"""
run_experiments_single.py
=========================
Single-seed oracle-in-the-loop PBO experiment for PI controller tuning.
Use this to verify the pipeline before running the full multi-seed version.

The run_loop() and all helper functions defined here are imported by
run_experiments.py — keep them self-contained and parameter-driven.

Quick-change section
--------------------
Edit only the three constants at the top to switch scenarios and AcqF:
    YAML_CONFIG_PATH  — which scenario YAML to load
    ACQF_TYPE         — acquisition function
    ACQF_BETA         — UCB exploration coefficient (ignored for EI/SR)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import yaml
from pathlib import Path
from dotenv import load_dotenv

from botorch.models import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import (
    qSimpleRegret,
    qExpectedImprovement,
    qUpperConfidenceBound,
    qLogExpectedImprovement,
)

from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
try:
    from botorch.sampling.normal import SobolQMCNormalSampler
except Exception:
    from botorch.sampling import SobolQMCNormalSampler

from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.acquisition.objective import LinearMCObjective

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

import control_utils as cu
from L2Gengine import L2GEngine
from oracle import Oracle, OraclePersona, compute_ground_truth_optimum

# Analysis plotting functions (panel-level, invoked from plot_results_*)
from plot_regret import plot_regret as _plot_regret_panel
from plot_heatmap import plot_heatmap_panel as _plot_heatmap_panel

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
load_dotenv()

# ============================================================
# QUICK-CHANGE SECTION — edit these three lines to switch runs
# ============================================================
YAML_CONFIG_PATH = Path(__file__).parent / "configs" / "s6_direction_only.yaml"   # ← swap to any scenario YAML
ACQF_TYPE        = "qUCB"               # "qUCB" | "qEI" | "qLogEI" | "qSR"
ACQF_BETA        = 3.0                  # UCB beta (ignored for qEI / qLogEI / qSR)
# ============================================================

# L2G context (PI controller task — shared across all scenarios)
CONTEXT = {
    "task": "PI Controller Tuning for Step Response",
    "parameters": ["Kp", "Ki"],
    "subfunctions": ["overshoot_pct", "settling_time", "tracking_mse"],
    "subfunction_descriptions": {
        "overshoot_pct": (
            "Percentage by which the response exceeds the reference (0-100). "
            "Physical_meaning: Measures stability and damping. "
            "Usually lower the better."
        ),
        "settling_time": (
            "Time in seconds to stay within 2% of reference. "
            "Physical_meaning: Measures speed of response. "
            "Usually lower the better."
        ),
        "tracking_mse": (
            "Mean Squared Error of tracking. "
            "Physical_meaning: General measure of error energy. "
            "Usually lower the better."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Config + setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_oracle(cfg: dict, seed: int, verbose: bool = True) -> Oracle:
    """Instantiate an Oracle from a scenario YAML config."""
    o_cfg = cfg["oracle"]
    persona = OraclePersona[cfg["persona"]]
    late_switch = o_cfg.get("late_switch_iter") or 4
    return Oracle(
        persona=persona,
        noise_level=o_cfg["noise_level"],
        seed=seed,
        late_switch_iter=late_switch,
        verbose=verbose,
    )


def build_plant_and_bounds(cfg: dict):
    """Return (plant, bounds_tensor) from YAML config."""
    p = cfg["plant"]
    plant = cu.make_plant(wn=p["wn"], zeta=p["zeta"])
    b = cfg["bounds"]
    bounds = torch.tensor(
        [[b["Kp"][0], b["Ki"][0]], [b["Kp"][1], b["Ki"][1]]],
        dtype=torch.double,
    )
    return plant, bounds


# ─────────────────────────────────────────────────────────────────────────────
# Simulation + surrogate helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidate(x_tensor, plant, t_final: float = 40.0):
    kp = float(x_tensor[0, 0].item())
    ki = float(x_tensor[0, 1].item())
    t, y = cu.simulate_step(kp, ki, plant, t_final=t_final)
    metrics = cu.compute_metrics(t, y)
    if np.isnan(metrics["settling_time"]):
        metrics["settling_time"] = t_final + 10.0
    if metrics["tracking_mse"] > 1e4 or np.isnan(metrics["tracking_mse"]):
        metrics["tracking_mse"] = 1e5
    return t, y, metrics


def update_surrogates(X_train, Y_dict: dict) -> dict:
    """Train one GP per metric. Returns dict of fitted SingleTaskGP models."""
    models = {}
    for name, Y_data in Y_dict.items():
        train_Y = (Y_data - Y_data.mean()) / (Y_data.std() + 1e-6)
        gp = SingleTaskGP(X_train, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models[name] = gp
    return models


def create_surrogate_wrapper(model, mean_val, std_val):
    """Return a de-normalising callable over the GP model."""
    def wrapper(X):
        ndim = X.dim()
        if ndim == 1:
            X_2d = X.unsqueeze(0)
        elif ndim == 3:
            X_2d = X.reshape(-1, X.shape[-1])
        else:
            X_2d = X
        val = model.posterior(X_2d).mean * std_val + mean_val
        if ndim == 1:
            return val.squeeze()
        elif ndim == 3:
            return val.reshape(X.shape[0], X.shape[1])
        return val.squeeze(-1)
    return wrapper


def warm_start(plant, bounds, n_samples: int, t_final: float = 40.0):
    """
    Evaluate n_samples Sobol candidates silently.
    Returns train_x, train_y (per metric), obs_metrics, obs_trajectories, best_idx.
    """
    print(f"--- WARM START: evaluating {n_samples} Sobol samples ---")
    raw_x = draw_sobol_samples(bounds=bounds, n=1, q=n_samples).squeeze(0).to(torch.double)

    obs_metrics = {"overshoot_pct": [], "settling_time": [], "tracking_mse": []}
    obs_trajectories = []
    valid_indices = []

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  {i}/{n_samples}...", end="\r")
        t, y, m = evaluate_candidate(raw_x[i].unsqueeze(0), plant, t_final)
        for k in obs_metrics:
            obs_metrics[k].append(m[k])
        obs_trajectories.append((t, y, m))
        if m["tracking_mse"] < 100.0 and m["settling_time"] < t_final - 1.0:
            valid_indices.append(i)

    print(f"  Done — {len(valid_indices)}/{n_samples} stable candidates found.")

    train_y = {k: torch.tensor(v).unsqueeze(-1).double() for k, v in obs_metrics.items()}

    if valid_indices:
        best_idx = valid_indices[int(np.argmin([obs_metrics["tracking_mse"][i] for i in valid_indices]))]
    else:
        print("  WARNING: No stable controllers found. Check bounds / plant.")
        best_idx = int(np.argmin(obs_metrics["tracking_mse"]))

    return raw_x, train_y, obs_metrics, obs_trajectories, best_idx


def build_acqf(acqf_type: str, pref_model, mse_model, acqf_beta: float):
    """
    Build the acquisition function.

    qUCB / qSR  — use pref_model when available (leverages preference feedback),
                  fall back to mse_model on cold start.
    qEI / qLogEI — always use mse_model (EI is defined over a scalar objective;
                   the MSE surrogate serves as a proxy throughout).
    """
    if acqf_type == "qUCB":
        model = pref_model if pref_model is not None else mse_model
        return qUpperConfidenceBound(model, beta=acqf_beta)

    if acqf_type == "qSR":
        model = pref_model if pref_model is not None else mse_model
        return qSimpleRegret(model)

    if acqf_type in ("qEI", "qLogEI"):
        y_norm = mse_model.train_targets
        best_f = y_norm.min()
        negate = LinearMCObjective(weights=torch.tensor([-1.0], dtype=torch.double))
        if acqf_type == "qEI":
            return qExpectedImprovement(model=mse_model, best_f=-best_f, objective=negate)
        return qLogExpectedImprovement(model=mse_model, best_f=-best_f, objective=negate)

    if acqf_type == "qEUBO":
        if pref_model is None:                      
            best_f = mse_model.train_targets.min()
            negate = LinearMCObjective(weights=torch.tensor([-1.0], dtype=torch.double))
            return qLogExpectedImprovement(model=mse_model, best_f=-best_f, objective=negate)
        with torch.no_grad():
            mean = pref_model.posterior(pref_model.datapoints).mean.reshape(-1)
        winner = pref_model.datapoints[int(torch.argmax(mean))].detach()
        return AnalyticExpectedUtilityOfBestOption(pref_model=pref_model, previous_winner=winner)

    if acqf_type == "qTS":
        model = pref_model if pref_model is not None else mse_model
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        return qSimpleRegret(model, sampler=sampler)

    raise ValueError(f"Unknown ACQF_TYPE '{acqf_type}'. Choose: qUCB | qEI | qLogEI | qSR")

def check_real_constraints(candidate_metrics: dict, specs: list, ref_metrics_dict: dict):
    """Verify real simulated metrics satisfy the L2G constraint specs."""
    for spec in specs:
        val = candidate_metrics.get(spec.subfunction_id)
        if val is None or np.isnan(val):
            return False, f"Metric {spec.subfunction_id} is NaN."
        limit = None
        if spec.constraint_type == "upper_bound_absolute":
            limit = spec.threshold
        elif spec.constraint_type in ("upper_bound_relative", "directional_improvement"):
            ref_key = spec.reference_candidate
            if ref_key not in ref_metrics_dict:
                continue
            ref_val = ref_metrics_dict[ref_key].get(spec.subfunction_id)
            if ref_val is None or np.isnan(ref_val):
                return False, f"Reference {ref_key} has NaN {spec.subfunction_id}"
            limit = ref_val + spec.margin if spec.operator == "<=" else ref_val - spec.margin
        if limit is not None:
            if spec.operator == "<=" and val > limit:
                return False, f"{spec.subfunction_id} ({val:.2f}) > limit {limit:.2f}"
            if spec.operator == ">=" and val < limit:
                return False, f"{spec.subfunction_id} ({val:.2f}) < limit {limit:.2f}"
    return True, "OK"


def _deduplicate_constraint_specs(l2g) -> None:
    """
    Remove duplicate constraint specs from the L2G engine's constraint list.

    Constrained oracles (DRASTIC_ABSOLUTE, DRASTIC_RELATIVE, CONTRADICTORY)
    re-emit the same constraint every iteration.  Without deduplication, the
    list grows to length n_iters, causing identical nonlinear constraints to be
    applied repeatedly — this is both redundant (no new information) and
    expensive (the optimiser must satisfy 10+ copies of the same bound).

    Deduplication is keyed on (subfunction_id, constraint_type, threshold,
    operator) so that semantically distinct constraints from different
    iterations are still preserved.
    """
    if not l2g.constraint_specs:
        return
    seen: set = set()
    unique: list = []
    for spec in l2g.constraint_specs:
        key = (
            spec.subfunction_id,
            spec.constraint_type,
            round(spec.threshold or 0.0, 8),
            spec.operator,
        )
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    l2g.constraint_specs = unique


# ─────────────────────────────────────────────────────────────────────────────
# Core optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(cfg: dict, oracle: Oracle, plant, bounds, U_star: float,
             acqf_type: str, acqf_beta: float, seed: int):
    """
    Run one complete PBO optimisation loop with the oracle as the feedback source.

    Parameters
    ----------
    cfg       : loaded YAML config dict
    oracle    : Oracle instance (persona + seed already set)
    plant     : control system plant
    bounds    : torch.Tensor [2, d]
    U_star    : ground-truth utility maximum (for regret computation)
    acqf_type : acquisition function type string
    acqf_beta : UCB beta
    seed      : RNG seed for this run

    Returns
    -------
    regrets      : list[float]        — simple regret after each iteration
    best_traj_x  : list[(Kp, Ki)]     — best (Kp, Ki) after each iteration
    train_x      : Tensor             — final training set
    obs_traj     : list               — final trajectory list
    best_idx     : int                — index of final best in train_x
    """
    set_all_seeds(seed)

    n_iters   = cfg["experiment"]["n_iterations"]
    n_warm    = cfg["experiment"]["warm_start_n"]
    n_retries = cfg["experiment"]["n_retries"]

    # ── Warm start ──────────────────────────────────────────────────────────
    train_x, train_y, obs_metrics, obs_traj, best_idx = warm_start(plant, bounds, n_warm)

    # L2G engine — no LLM needed; oracle output is passed as simulated_output
    l2g = L2GEngine(context=CONTEXT, sub_surrogates={}, llm=None)

    last_shown_x       = None
    last_shown_metrics = None
    regrets            = []
    best_traj_x        = []

    # ── Main loop ────────────────────────────────────────────────────────────
    for iteration in range(1, n_iters + 1):
        print(f"\n=== ITERATION {iteration}/{n_iters} ===")

        # 1. Train metric surrogates
        metric_models = update_surrogates(train_x, train_y)

        # 2. Wrap surrogates for L2G (de-normalised outputs)
        l2g_surrogates = {}
        for k, gp in metric_models.items():
            std_val  = train_y[k].std() + 1e-6
            mean_val = train_y[k].mean()
            l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
        l2g.sub_surrogates = l2g_surrogates

        # 3. Train preference model (PairwiseGP) if comparisons are available
        pref_model = None
        pref_data  = l2g.get_preference_dataset()
        if pref_data:
            print(f"  Training preference model on {len(pref_data)} comparisons...")
            X_win   = torch.cat([p[0] for p in pref_data], dim=0)
            X_lose  = torch.cat([p[1] for p in pref_data], dim=0)
            train_X = torch.cat([X_win, X_lose], dim=0)
            M       = X_win.shape[0]
            comps   = torch.tensor([[i, M + i] for i in range(M)], dtype=torch.long)
            pref_model = PairwiseGP(datapoints=train_X, comparisons=comps)
            fit_gpytorch_mll(
                PairwiseLaplaceMarginalLogLikelihood(pref_model.likelihood, pref_model)
            )

        # 4. Deduplicate constraint specs before building constraints.
        #    Constrained oracles re-emit the same constraint each iteration;
        #    accumulation (without dedup) shrinks the feasible set to near-zero
        #    even though the constraint itself hasn't changed.
        _deduplicate_constraint_specs(l2g)

        cand_ref = {"A": train_x[best_idx]}
        if last_shown_x is not None:
            cand_ref["B"] = last_shown_x.squeeze(0)
        constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)
        if constraints_funcs:
            print(f"  Applying {len(constraints_funcs)} active language constraints.")

        # 5. Build acquisition function
        mse_model = metric_models["tracking_mse"]
        acq_func  = build_acqf(acqf_type, pref_model, mse_model, acqf_beta)

        # 6. Reference metrics for real constraint verification
        metricsA    = obs_traj[best_idx][2]
        ref_metrics = {"A": metricsA}
        if last_shown_metrics is not None:
            ref_metrics["B"] = last_shown_metrics

        # 7. Optimise acqf with retry / active-learning loop
        found_valid  = False
        new_candidate = None
        metricsB      = None
        tB = yB       = None

        for attempt in range(n_retries):
            # A. Propose candidate
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

            # B. Simulate
            tB, yB, metricsB = evaluate_candidate(new_candidate, plant)

            # C. Verify real constraints
            is_valid, reason = check_real_constraints(metricsB, l2g.constraint_specs, ref_metrics)
            if is_valid:
                found_valid = True
                print(f"  Attempt {attempt+1}: valid candidate found.")
                break

            # D. Active learning — add failed point and refit
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
            # Re-deduplicate after active-learning refit (new specs may have been added)
            _deduplicate_constraint_specs(l2g)
            constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)

        if not found_valid:
            print("  WARNING: No valid candidate after all retries — using last proposal.")

        # 8. Oracle feedback (replaces human input)
        xA = train_x[best_idx]
        xB = new_candidate.squeeze(0)
        oracle_output = oracle.query(metricsA, metricsB, iteration=iteration)

        # 9. Process oracle output through L2G (updates preference history + constraints)
        parsed = l2g.process_feedback(
            candidates={"A": xA, "B": xB},
            feedback_text="",
            simulated_output=oracle_output,
        )

        # 10. Add successful candidate to training data (if not already added in retry loop)
        if not torch.equal(train_x[-1], new_candidate):
            train_x = torch.cat([train_x, new_candidate], dim=0)
            for k in obs_metrics:
                train_y[k] = torch.cat([train_y[k], torch.tensor([[metricsB[k]]], dtype=torch.double)], dim=0)
                obs_metrics[k].append(metricsB[k])
            obs_traj.append((tB, yB, metricsB))

        # 11. Update best_idx based on oracle preference (or utility comparison if no preference given)
        idx_A = best_idx
        idx_B = len(train_x) - 1

        preferred = None
        if parsed and parsed.preference:
            preferred = parsed.preference.preferred_candidate

        if preferred == "B":
            best_idx = idx_B
            print("  → Baseline updated to B.")
        elif preferred is None and found_valid:
            # Direction-only (and any no-preference path): fall back to oracle utility comparison.
            # Without this, best_idx never moves and regret is constant for all iterations.
            util_A = oracle.utility(metricsA)
            util_B = oracle.utility(metricsB)
            if util_B > util_A:
                best_idx = idx_B
                print(f"  → No preference; U(B)={util_B:.4f} > U(A)={util_A:.4f} — baseline updated to B.")
            else:
                best_idx = idx_A
                print(f"  → No preference; U(A)={util_A:.4f} >= U(B)={util_B:.4f} — baseline unchanged.")
        else:
            best_idx = idx_A
            if preferred == "A":
                print("  → Baseline remains A.")
            else:
                n_new = len(parsed.constraints) if parsed else 0
                print(f"  → No preference; baseline remains A ({n_new} new constraints).")

        # 12. Track simple regret
        current_metrics = obs_traj[best_idx][2]
        current_utility = oracle.utility(current_metrics)
        regret          = U_star - current_utility
        regrets.append(regret)

        kp_best = train_x[best_idx][0].item()
        ki_best = train_x[best_idx][1].item()
        best_traj_x.append((kp_best, ki_best))

        print(
            f"  Best: Kp={kp_best:.3f}, Ki={ki_best:.3f} | "
            f"U={current_utility:.4f} | Regret={regret:.4f}"
        )

    return regrets, best_traj_x, train_x, obs_traj, best_idx


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — single seed
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_single(
    regrets, best_traj_x, grid_data: dict, cfg: dict,
    acqf_type: str, seed: int, U_star: float,
):
    """
    Two-panel figure assembled from analysis-folder panel functions:
      Left  — simple regret per iteration  (plot_regret.plot_regret)
      Right — oracle utility heatmap + trajectory  (plot_heatmap.plot_heatmap_panel)

    Saves to cfg["logging"]["output_dir"] and closes the figure without
    blocking the terminal (plt.show() is intentionally omitted).
    """
    persona_name = cfg["persona"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"PBO  ·  {persona_name}  ·  {acqf_type}  ·  seed {seed}",
        fontsize=13,
    )

    # Left panel — regret curve with convergence annotation
    _plot_regret_panel(
        np.array(regrets),
        u_star=U_star,
        title="Simple Regret",
        ax=ax1,
        annotate_convergence=True,
    )

    # Right panel — utility heatmap with best-so-far trajectory
    _plot_heatmap_panel(
        grid_data,
        best_traj_x,
        title="Utility Landscape + Trajectory",
        seed_label=f"seed {seed}",
        ax=ax2,
    )

    plt.tight_layout()

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"single_seed{seed}_{acqf_type}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {fname}")
    plt.close("all")   # avoids blocking the terminal (no plt.show())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg           = load_config(YAML_CONFIG_PATH)
    plant, bounds = build_plant_and_bounds(cfg)
    seed          = cfg["experiment"]["seeds"][0]

    set_all_seeds(seed)

    # Build oracle — verbose for single-seed run
    oracle = build_oracle(cfg, seed=seed, verbose=True)

    # Ground-truth optimum (noise-free utility, dense grid)
    print("Computing ground truth optimum (this may take ~30–90 s) ...")
    oracle_gt = build_oracle(cfg, seed=seed, verbose=False)
    oracle_gt.noise_level = 0.0
    best_params, U_star, grid_data = compute_ground_truth_optimum(
        plant, bounds.numpy(), oracle_gt, n_grid=30
    )
    print(f"U* = {U_star:.4f}  at  Kp*={best_params[0]:.3f}, Ki*={best_params[1]:.3f}")

    # Run optimisation loop
    regrets, best_traj_x, train_x, obs_traj, best_idx = run_loop(
        cfg, oracle, plant, bounds, U_star, ACQF_TYPE, ACQF_BETA, seed
    )

    # Plot
    plot_results_single(regrets, best_traj_x, grid_data, cfg, ACQF_TYPE, seed, U_star)

    # Final summary
    print("\n=== DONE ===")
    fm = obs_traj[best_idx][2]
    print(
        f"Final best: Kp={train_x[best_idx][0]:.3f}, Ki={train_x[best_idx][1]:.3f}\n"
        f"  Overshoot  = {fm['overshoot_pct']:.2f}%\n"
        f"  Settling   = {fm['settling_time']:.2f} s\n"
        f"  MSE        = {fm['tracking_mse']:.4f}\n"
        f"  Final regret = {regrets[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
