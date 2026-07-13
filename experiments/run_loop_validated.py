"""
run_loop_validated.py
=====================
ONE loop, THREE arms — the experiment core for the validator study.

    arm="pbo"        Standard PBO: preference only.  Constraints are stripped
                     from the oracle output before L2G sees them.
    arm="l2g"        PBO + L2G: preferences + language constraints — the
                     unchanged baseline behaviour of run_experiments_single.
    arm="validated"  PBO + L2G + Validator: same as "l2g", plus the
                     PreferenceValidator audits the comparison history and
                     occasionally converts an iteration into a VALIDATION
                     DUEL (re-shows a stored pair to the oracle, order-swapped,
                     zero new simulations).  See validator.py.

Design rules honoured:
  • every arm spends exactly ONE oracle query per iteration (fair regret-vs-
    iteration comparison);
  • no preference is ever deleted — only reweighted (replication);
  • regret is DYNAMIC: computed against the oracle's currently active utility
    via `u_star_fn(iteration)` and `oracle.utility_at(metrics, iteration)`
    (falls back to the static `oracle.utility` for non-switching personas).

All shared helpers are imported unchanged from run_experiments_single, same
pattern as run_loop_robust.py.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions

from run_experiments_single import (
    CONTEXT,
    set_all_seeds,
    evaluate_candidate,
    update_surrogates,
    create_surrogate_wrapper,
    warm_start,
    build_acqf,
    check_real_constraints,
    _deduplicate_constraint_specs,
)
from L2Gengine import L2GEngine
from validator import PreferenceValidator

ARMS = ("pbo", "l2g", "validated")


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utility_now(oracle, metrics: dict, iteration: int) -> float:
    """Utility under the oracle's currently active regime (dynamic regret)."""
    if hasattr(oracle, "utility_at"):
        return oracle.utility_at(metrics, iteration)
    return oracle.utility(metrics)


def _strip_constraints(oracle_output: dict) -> dict:
    """Arm 'pbo': keep only the preference channel of the oracle output."""
    out = dict(oracle_output)
    out["constraints"] = []
    if out.get("feedback_type") == "both":
        out["feedback_type"] = "preference_only"
    elif out.get("feedback_type") == "direction_only":
        out["feedback_type"] = "none"
    return out


def _fit_pref_gp(pref_pairs) -> Optional[PairwiseGP]:
    """Plain (baseline-identical) PairwiseGP fit on (x_win, x_lose) pairs."""
    if not pref_pairs:
        return None
    X_win = torch.cat([p[0] for p in pref_pairs], dim=0)
    X_lose = torch.cat([p[1] for p in pref_pairs], dim=0)
    train_X = torch.cat([X_win, X_lose], dim=0)
    M = X_win.shape[0]
    comps = torch.tensor([[i, M + i] for i in range(M)], dtype=torch.long)
    model = PairwiseGP(datapoints=train_X, comparisons=comps)
    fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    return model


def _posterior_argmax(pref_model, train_x, fallback: int) -> int:
    """Incumbent fallback: argmax of the preference posterior over all points."""
    if pref_model is None:
        return int(fallback)
    with torch.no_grad():
        mean = pref_model.posterior(train_x).mean.reshape(-1)
    return int(torch.argmax(mean).item())


def _ensure_2d(x):
    return x.unsqueeze(0) if x.dim() == 1 else x


# ─────────────────────────────────────────────────────────────────────────────
# the loop
# ─────────────────────────────────────────────────────────────────────────────

def run_loop_validated(
    cfg: dict,
    oracle,
    plant,
    bounds,
    u_star_fn: Callable[[int], float],
    seed: int,
    *,
    arm: str = "validated",
    acqf_type: str = "qEUBO",
    acqf_beta: float = 2.0,
    validator_kwargs: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Returns
    -------
    regrets, best_traj_x, train_x, obs_traj, best_idx, extras
        extras = {"validation_iters": [...], "events": [...], "summary": {...}}
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got '{arm}'")

    set_all_seeds(seed)

    n_iters = cfg["experiment"]["n_iterations"]
    n_warm = cfg["experiment"]["warm_start_n"]
    n_retries = cfg["experiment"]["n_retries"]

    train_x, train_y, obs_metrics, obs_traj, best_idx = warm_start(plant, bounds, n_warm)

    l2g = L2GEngine(context=CONTEXT, sub_surrogates={}, llm=None)
    validator = (PreferenceValidator(**(validator_kwargs or {}))
                 if arm == "validated" else None)

    last_shown_x = None
    last_shown_metrics = None
    regrets = []
    best_traj_x = []
    validation_iters = []

    for iteration in range(1, n_iters + 1):
        if verbose:
            print(f"\n=== [{arm.upper()}] ITERATION {iteration}/{n_iters} ===")

        # 1-2. Metric surrogates + L2G wrappers
        metric_models = update_surrogates(train_x, train_y)
        l2g_surrogates = {}
        for k, gp in metric_models.items():
            std_val = train_y[k].std() + 1e-6
            mean_val = train_y[k].mean()
            l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
        l2g.sub_surrogates = l2g_surrogates

        # 3. Preference model — validator arm fits on the WEIGHTED dataset,
        #    other arms on the raw L2G history (baseline-identical).
        pref_pairs = (validator.dataset() if arm == "validated"
                      else l2g.get_preference_dataset())
        pref_model = _fit_pref_gp(pref_pairs)

        # ── VALIDATION BRANCH ────────────────────────────────────────────────
        # Audit the history; if a comparison is flagged, eligible (retest_delay)
        # and in budget, THIS iteration becomes a validation duel: same pair,
        # order-swapped, no acqf, no simulation.  One oracle query — same cost
        # as a normal iteration.
        did_validation = False
        if arm == "validated":
            rec = validator.audit(pref_model, iteration, refit_fn=_fit_pref_gp)
            if rec is not None:
                did_validation = True
                validation_iters.append(iteration)
                if verbose:
                    print(f"  [validator] VALIDATION DUEL on rec#{rec.rec_id} "
                          f"(expressed it={rec.iter_added}, p={rec.last_p:.3f})")
                # Order swap: A = stored loser, B = stored winner.
                retest_out = oracle.query(rec.m_lose, rec.m_win, iteration=iteration)
                pref = retest_out.get("preference")
                if pref is not None:
                    confirmed = pref["preferred_candidate"] == "B"   # B = orig winner
                    validator.resolve(rec, confirmed, iteration)
                    # The end-of-iteration refit + posterior-argmax incumbent
                    # (step 11) absorbs the correction/reweighting immediately.
                # No preference in the retest output (should not happen with
                # these personas): leave the record pending; audit will retry.

        # ── NORMAL BRANCH ────────────────────────────────────────────────────
        if not did_validation:
            # 4. Language constraints (arm 'pbo' never accumulates specs, so
            #    this is a no-op there).
            _deduplicate_constraint_specs(l2g)
            cand_ref = {"A": train_x[best_idx]}
            if last_shown_x is not None:
                cand_ref["B"] = last_shown_x.squeeze(0)
            constraints_funcs = l2g.build_nonlinear_inequality_constraints(
                candidates_reference=cand_ref)
            if constraints_funcs and verbose:
                print(f"  Applying {len(constraints_funcs)} active language constraints.")

            # 5. Acquisition function (qEUBO by default)
            mse_model = metric_models["tracking_mse"]
            acq_func = build_acqf(acqf_type, pref_model, mse_model, acqf_beta)

            # 6. Reference metrics for real-constraint verification
            metricsA = obs_traj[best_idx][2]
            ref_metrics = {"A": metricsA}
            if last_shown_metrics is not None:
                ref_metrics["B"] = last_shown_metrics

            # 7. Optimise acqf with retry / active-learning loop
            found_valid = False
            new_candidate = None
            metricsB = None
            tB = yB = None

            for attempt in range(n_retries):
                try:
                    new_candidate, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=2048,
                        nonlinear_inequality_constraints=constraints_funcs or None,
                        ic_generator=gen_batch_initial_conditions,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  Attempt {attempt+1}: optimiser error ({e}). Random fallback.")
                    new_candidate = draw_sobol_samples(
                        bounds=bounds, n=1, q=1).squeeze(0).to(torch.double)

                tB, yB, metricsB = evaluate_candidate(new_candidate, plant)
                is_valid, reason = check_real_constraints(
                    metricsB, l2g.constraint_specs, ref_metrics)
                if is_valid:
                    found_valid = True
                    break

                if verbose:
                    print(f"  Attempt {attempt+1}: failed ({reason}). Active refit...")
                train_x = torch.cat([train_x, new_candidate], dim=0)
                for k in obs_metrics:
                    train_y[k] = torch.cat(
                        [train_y[k], torch.tensor([[metricsB[k]]], dtype=torch.double)], dim=0)
                    obs_metrics[k].append(metricsB[k])
                obs_traj.append((tB, yB, metricsB))

                metric_models = update_surrogates(train_x, train_y)
                for k, gp in metric_models.items():
                    std_val = train_y[k].std() + 1e-6
                    mean_val = train_y[k].mean()
                    l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
                l2g.sub_surrogates = l2g_surrogates
                _deduplicate_constraint_specs(l2g)
                constraints_funcs = l2g.build_nonlinear_inequality_constraints(
                    candidates_reference=cand_ref)

            if not found_valid and verbose:
                print("  WARNING: No valid candidate after all retries — using last proposal.")

            # 8. Oracle feedback on the fresh duel A (incumbent) vs B (new)
            xA = train_x[best_idx]
            xB = new_candidate.squeeze(0)
            oracle_output = oracle.query(metricsA, metricsB, iteration=iteration)

            # 9. Route through L2G — arm 'pbo' gets the constraint-stripped view.
            l2g_input = (_strip_constraints(oracle_output) if arm == "pbo"
                         else oracle_output)
            parsed = l2g.process_feedback(
                candidates={"A": xA, "B": xB},
                feedback_text="",
                simulated_output=l2g_input,
            )

            # 10. Add the evaluated candidate to training data
            if not torch.equal(train_x[-1], new_candidate):
                train_x = torch.cat([train_x, new_candidate], dim=0)
                for k in obs_metrics:
                    train_y[k] = torch.cat(
                        [train_y[k], torch.tensor([[metricsB[k]]], dtype=torch.double)], dim=0)
                    obs_metrics[k].append(metricsB[k])
                obs_traj.append((tB, yB, metricsB))

            # 10b. Register the comparison with the validator (never deleted).
            preferred = parsed.preference.preferred_candidate if (
                parsed and parsed.preference) else None
            if arm == "validated" and preferred in ("A", "B"):
                if preferred == "A":
                    validator.add_comparison(
                        _ensure_2d(xA), _ensure_2d(xB), metricsA, metricsB, iteration)
                else:
                    validator.add_comparison(
                        _ensure_2d(xB), _ensure_2d(xA), metricsB, metricsA, iteration)

            last_shown_x = xB.unsqueeze(0)
            last_shown_metrics = metricsB

        # 11. Incumbent = posterior argmax of the preference model, refit on
        #     the dataset as it stands at the END of this iteration.  Arm-
        #     neutral (replaces the "last preferred" rule): a single flipped
        #     label can no longer teleport the incumbent, and validator
        #     corrections / reweighting move the incumbent — hence the
        #     measured regret — in the same iteration they happen.
        pref_pairs = (validator.dataset() if arm == "validated"
                      else l2g.get_preference_dataset())
        pref_model = _fit_pref_gp(pref_pairs)
        best_idx = _posterior_argmax(pref_model, train_x, best_idx)

        # 12. Dynamic simple regret vs the CURRENTLY ACTIVE utility
        current_metrics = obs_traj[best_idx][2]
        current_u = _utility_now(oracle, current_metrics, iteration)
        regret = u_star_fn(iteration) - current_u
        regrets.append(regret)

        kp_best = train_x[best_idx][0].item()
        ki_best = train_x[best_idx][1].item()
        best_traj_x.append((kp_best, ki_best))
        if verbose:
            tag = "VAL" if did_validation else "opt"
            print(f"  [{tag}] Best: Kp={kp_best:.3f}, Ki={ki_best:.3f} | "
                  f"U={current_u:.4f} | Regret={regret:.4f}")

    extras = {
        "validation_iters": validation_iters,
        "events": validator.event_log if validator else [],
        "summary": validator.summary() if validator else {},
    }
    return regrets, best_traj_x, train_x, obs_traj, best_idx, extras
