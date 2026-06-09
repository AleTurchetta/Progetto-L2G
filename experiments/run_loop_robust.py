"""
run_loop_robust.py
==================
A SEPARATE, robust variant of run_experiments_single.run_loop.

Nothing in run_experiments_single is modified.  Every helper
(warm_start, update_surrogates, create_surrogate_wrapper, evaluate_candidate,
build_acqf, check_real_constraints, _deduplicate_constraint_specs, CONTEXT) is
IMPORTED from there unchanged — only two steps differ, and both are tagged with
`# === ROBUST CHANGE ===` so the diff vs the original is obvious:

  • Step 3  — preference model: clean comparisons (trust-but-verify vs the metric
              surrogates + contradiction de-dup) and fit a noise-tolerant
              PairwiseGP, via robust_pref.build_robust_pref_model.
  • Step 11 — incumbent: argmax of the preference posterior over ALL evaluated
              points, via robust_pref.robust_incumbent, instead of
              "best_idx = whoever was preferred last".

"""

from __future__ import annotations

import torch

from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions

# Unchanged helpers reused from the original single-seed script.
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

import robust_pref


def run_loop_robust(
    cfg: dict, oracle, plant, bounds, U_star: float,
    acqf_type: str, acqf_beta: float, seed: int,
    *,
    grounding: str = "utility",   # "utility" (oracle-informed ceiling) | "pareto" (deployable) | "off"
    pref_jitter: float = 1e-2,    # noise tolerance of the PairwiseGP
    dedup: bool = True,
):
    """
    Robust PBO optimisation loop.  Same contract as run_experiments_single.run_loop.

    Returns
    -------
    regrets, best_traj_x, train_x, obs_traj, best_idx
    """
    set_all_seeds(seed)

    n_iters   = cfg["experiment"]["n_iterations"]
    n_warm    = cfg["experiment"]["warm_start_n"]
    n_retries = cfg["experiment"]["n_retries"]

    # ── Warm start (identical to the baseline → same start across personas) ──
    train_x, train_y, obs_metrics, obs_traj, best_idx = warm_start(plant, bounds, n_warm)

    l2g = L2GEngine(context=CONTEXT, sub_surrogates={}, llm=None)

    last_shown_x       = None
    last_shown_metrics = None
    regrets            = []
    best_traj_x        = []

    for iteration in range(1, n_iters + 1):
        print(f"\n=== [ROBUST] ITERATION {iteration}/{n_iters} ===")

        # 1. Metric surrogates
        metric_models = update_surrogates(train_x, train_y)

        # 2. Wrap surrogates for L2G (de-normalised outputs)
        l2g_surrogates = {}
        for k, gp in metric_models.items():
            std_val  = train_y[k].std() + 1e-6
            mean_val = train_y[k].mean()
            l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
        l2g.sub_surrogates = l2g_surrogates

        # 3. Preference model
        # === ROBUST CHANGE ===  clean comparisons (trust-but-verify + de-dup),
        #     then fit a noise-tolerant PairwiseGP.
        pref_model = robust_pref.build_robust_pref_model(
            l2g.get_preference_dataset(),
            surrogate_wrappers=l2g.sub_surrogates,        # uncorrupted metric channel
            utility=getattr(oracle, "utility_fn", None),  # used only if grounding="utility"
            grounding=grounding,
            jitter=pref_jitter,
            dedup=dedup,
            verbose=False,
        )
        # === END ROBUST CHANGE ===

        # 4. Build language constraints
        _deduplicate_constraint_specs(l2g)
        cand_ref = {"A": train_x[best_idx]}
        if last_shown_x is not None:
            cand_ref["B"] = last_shown_x.squeeze(0)
        constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)
        if constraints_funcs:
            print(f"  Applying {len(constraints_funcs)} active language constraints.")

        # 5. Acquisition function
        mse_model = metric_models["tracking_mse"]
        acq_func  = build_acqf(acqf_type, pref_model, mse_model, acqf_beta)

        # 6. Reference metrics for real constraint verification
        metricsA    = obs_traj[best_idx][2]
        ref_metrics = {"A": metricsA}
        if last_shown_metrics is not None:
            ref_metrics["B"] = last_shown_metrics

        # 7. Optimise acqf with retry / active-learning loop
        found_valid   = False
        new_candidate = None
        metricsB      = None
        tB = yB       = None

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
                print(f"  Attempt {attempt+1}: optimiser error ({e}). Random fallback.")
                new_candidate = draw_sobol_samples(bounds=bounds, n=1, q=1).squeeze(0).to(torch.double)

            tB, yB, metricsB = evaluate_candidate(new_candidate, plant)

            is_valid, reason = check_real_constraints(metricsB, l2g.constraint_specs, ref_metrics)
            if is_valid:
                found_valid = True
                print(f"  Attempt {attempt+1}: valid candidate found.")
                break

            print(f"  Attempt {attempt+1}: failed ({reason}). Active refit...")
            train_x = torch.cat([train_x, new_candidate], dim=0)
            for k in obs_metrics:
                train_y[k] = torch.cat([train_y[k], torch.tensor([[metricsB[k]]], dtype=torch.double)], dim=0)
                obs_metrics[k].append(metricsB[k])
            obs_traj.append((tB, yB, metricsB))

            metric_models = update_surrogates(train_x, train_y)
            for k, gp in metric_models.items():
                std_val  = train_y[k].std() + 1e-6
                mean_val = train_y[k].mean()
                l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
            l2g.sub_surrogates = l2g_surrogates
            _deduplicate_constraint_specs(l2g)
            constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)

        if not found_valid:
            print("  WARNING: No valid candidate after all retries — using last proposal.")

        # 8. Oracle feedback
        xA = train_x[best_idx]
        xB = new_candidate.squeeze(0)
        oracle_output = oracle.query(metricsA, metricsB, iteration=iteration)

        # 9. Process oracle output through L2G
        parsed = l2g.process_feedback(
            candidates={"A": xA, "B": xB},
            feedback_text="",
            simulated_output=oracle_output,
        )

        # 10. Add successful candidate to training data (if not already added)
        if not torch.equal(train_x[-1], new_candidate):
            train_x = torch.cat([train_x, new_candidate], dim=0)
            for k in obs_metrics:
                train_y[k] = torch.cat([train_y[k], torch.tensor([[metricsB[k]]], dtype=torch.double)], dim=0)
                obs_metrics[k].append(metricsB[k])
            obs_traj.append((tB, yB, metricsB))

        # 11. Update incumbent
        # === ROBUST CHANGE ===  argmax of the preference posterior over ALL
        #     evaluated points (a single flip can't move the incumbent).
        #     NOTE: pref_model here was fit at step 3, so it reflects comparisons
        #     up to the previous iteration (one-iteration lag — harmless).
        best_idx = robust_pref.robust_incumbent(pref_model, train_x, fallback_idx=best_idx)
        print(f"  → Robust incumbent: idx={best_idx}")
        # === END ROBUST CHANGE ===

        last_shown_x       = xB.unsqueeze(0)
        last_shown_metrics = metricsB

        # 12. Track simple regret
        current_metrics = obs_traj[best_idx][2]
        current_utility = oracle.utility(current_metrics)
        regret          = U_star - current_utility
        regrets.append(regret)

        kp_best = train_x[best_idx][0].item()
        ki_best = train_x[best_idx][1].item()
        best_traj_x.append((kp_best, ki_best))

        print(f"  Best: Kp={kp_best:.3f}, Ki={ki_best:.3f} | "
              f"U={current_utility:.4f} | Regret={regret:.4f}")

    return regrets, best_traj_x, train_x, obs_traj, best_idx
