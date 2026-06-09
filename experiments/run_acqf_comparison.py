"""
run_acqf_comparison.py
======================
Acquisition-function comparison experiment.

For the chosen acquisition function(s) it runs the oracle-in-the-loop PBO for
THREE personas — the consistent base oracle plus two adversarial personas you
pick — over several seeds, and overlays their Simple-Regret curves on one graph
(one figure per acquisition function).  
Goal: see which acqf keeps the hard personas closest to the easy reference.

Design guarantees
-----------------
• ONE random (reachable) target is drawn at the start and SHARED by all three
  personas and all seeds, so the hidden objective (and U*) is identical — only
  the feedback channel differs.
• Warm start is identical across personas for a given seed (run_loop reseeds
  from `seed` before its Sobol warm start), giving a clean paired comparison.

Heavy libraries (torch / botorch / control) are imported lazily inside the
functions that need them, so this module imports — and its orchestration can be
unit-tested — with only numpy + matplotlib present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

# Make sibling modules importable regardless of where the script is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from base_oracle import (
    Persona,
    TargetSeekingUtility,
    make_oracle,
    generate_random_target,
    compute_ground_truth_optimum,
)
from compare_acqf_plots import (
    BASE_LABEL,
    plot_per_acqf_figures,
    plot_acqf_grid,
    save_per_acqf_figures,
    robustness_gap_table,
)
from convergence_utils import DEFAULT_EPS


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QUICK-CHANGE CONFIG — edit this block, then run `python run_acqf_comparison.py`  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# The two ADVERSARIAL personas to overlay against the base oracle.
# Choose any two from base_oracle.ADVERSARIAL_PERSONAS:
#   NOISY · CONTRADICTORY · DRASTIC_ABSOLUTE · DRASTIC_RELATIVE ·
#   DIRECTION_ONLY · LATE_SWITCHER · AMBIGUOUS_POSITIVE
PERSONA_A = "NOISY"
PERSONA_B = "LATE_SWITCHER"

# Acquisition function(s) to test.  One overlay figure is produced per entry.
#   "qUCB" | "qSR" | "qEI" | "qLogEI" | "qEUBO" | "qTS"

ACQF_TYPES = ["qUCB", "qSR", "qEI", "qLogEI", "qEUBO", "qTS"]
ACQF_BETA  = 2.0                     # UCB beta (ignored by the others)

# Experiment budget.
SEEDS         = [42, 7, 13, 99, 2025]
N_ITERATIONS  = 20
WARM_START_N  = 20
N_RETRIES     = 10

# Plant + search space (match your baseline so results are comparable).
PLANT_WN, PLANT_ZETA = 1.0, 0.7
BOUNDS_KP = (0.1, 5.0)
BOUNDS_KI = (0.001, 3.0)

# Standardized target-seeking utility.
TARGET_SEED   = 123                  # fixes the random target (reproducible)
UTILITY_WEIGHTS = (1.0, 1.0, 1.0)    # (overshoot, settling, mse) — same for all personas
GRID_N        = 30                   # density of the ground-truth grid for U*
EPS_CONV      = DEFAULT_EPS          # utility-gap convergence threshold (0.02)

# Optional absolute anchor (model-free random search) added as an extra curve.
INCLUDE_RANDOM_ANCHOR = False

# Output.
OUT_DIR = Path(__file__).parent / "experiments" / "results" / "acqf_compare"

# Per-persona constructor kwargs (only used if that persona is selected).
PERSONA_KWARGS: Dict[str, dict] = {
    "NOISY":         {"noise_level": 0.20},
    "LATE_SWITCHER": {"switch_iter": 4},
}

# ╚══════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers (lazy heavy imports)
# ─────────────────────────────────────────────────────────────────────────────

def build_plant_and_bounds():
    """Return (plant, bounds_tensor[2,2] double, bounds_np[2,2])."""
    import torch
    import control_utils as cu
    plant = cu.make_plant(wn=PLANT_WN, zeta=PLANT_ZETA)
    bounds = torch.tensor(
        [[BOUNDS_KP[0], BOUNDS_KI[0]], [BOUNDS_KP[1], BOUNDS_KI[1]]],
        dtype=torch.double,
    )
    return plant, bounds, bounds.numpy()


def build_shared_utility_and_gt(plant, bounds_np):
    """
    Draw ONE reachable target (shared by every persona/seed), build the utility,
    and compute the ground-truth optimum for regret + the heatmap.
    """
    target, target_params = generate_random_target(plant, bounds_np, seed=TARGET_SEED)
    utility = TargetSeekingUtility(target, weights=UTILITY_WEIGHTS)
    print(f"Random target  : ov={target['overshoot_pct']:.2f}%, "
          f"ts={target['settling_time']:.2f}s, mse={target['tracking_mse']:.4f} "
          f"(from Kp={target_params[0]:.3f}, Ki={target_params[1]:.3f})")
    print("Computing ground-truth optimum on the grid (this can take ~30-90 s)...")
    best_params, u_star, grid_data = compute_ground_truth_optimum(
        plant, bounds_np, utility, n_grid=GRID_N
    )
    print(f"U* = {u_star:.4f}  at  Kp*={best_params[0]:.3f}, Ki*={best_params[1]:.3f}")
    return utility, u_star, grid_data


def _make_cfg() -> dict:
    """Minimal cfg dict consumed by run_experiments_single.run_loop."""
    return {"experiment": {"n_iterations": N_ITERATIONS,
                           "warm_start_n": WARM_START_N,
                           "n_retries": N_RETRIES}}


# ─────────────────────────────────────────────────────────────────────────────
# Single-run wrapper around the existing optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(persona_label, acqf, seed, *, utility, plant, bounds, u_star, cfg):
    """
    One PBO run for (persona, acqf, seed).  Builds the persona on the SHARED
    utility and delegates the optimisation to the existing run_loop.

    Returns
    -------
    regrets : list[float]  — Simple regret per iteration (length N_ITERATIONS)
    """
    from run_experiments_single import run_loop  # lazy: pulls torch/botorch
    # NOTE: requires build_acqf in run_experiments_single.py to support the
    # chosen acqf (add the qEUBO / qTS branches as described).
    kwargs = PERSONA_KWARGS.get(persona_label, {})
    oracle = make_oracle(Persona(persona_label), utility, seed=seed,
                         verbose=False, **kwargs)
    regrets, _best_traj, _tx, _traj, _bi = run_loop(
        cfg, oracle, plant, bounds, u_star, acqf, ACQF_BETA, seed
    )
    return regrets


def random_search_regrets(seed, *, utility, plant, bounds_np, u_star):
    """
    Model-free absolute anchor: warm start, then keep drawing uniform random
    controllers, tracking best-so-far utility.  Returns regret per iteration.
    """
    import base_oracle as bo
    rng = np.random.default_rng(seed)
    lo, hi = bounds_np[0], bounds_np[1]

    def u_of(kp, ki):
        return utility.utility(bo._eval_metrics(float(kp), float(ki), plant))

    # Warm start best.
    best_u = -np.inf
    for _ in range(WARM_START_N):
        kp, ki = rng.uniform(lo, hi)
        best_u = max(best_u, u_of(kp, ki))
    # Iterations.
    regrets = []
    for _ in range(N_ITERATIONS):
        kp, ki = rng.uniform(lo, hi)
        best_u = max(best_u, u_of(kp, ki))
        regrets.append(u_star - best_u)
    return regrets

def build_results(
    acqf_types: Sequence[str],
    persona_labels: Sequence[str],
    run_single: Callable[[str, str, int], Sequence[float]],
    seeds: Sequence[int] = tuple(SEEDS),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Assemble {acqf: {persona_label: ndarray[n_seeds, n_iters]}} by calling
    `run_single(persona_label, acqf, seed)` for every combination.
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for acqf in acqf_types:
        print(f"\n{'#'*64}\n#  ACQF = {acqf}\n{'#'*64}")
        per_persona: Dict[str, np.ndarray] = {}
        for label in persona_labels:
            print(f"\n--- persona: {label} ---")
            curves = []
            for seed in seeds:
                print(f"  seed {seed} ...")
                curves.append(np.asarray(run_single(label, acqf, seed), dtype=float))
            per_persona[label] = np.vstack(curves)
        results[acqf] = per_persona
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    plant, bounds, bounds_np = build_plant_and_bounds()
    utility, u_star, _grid = build_shared_utility_and_gt(plant, bounds_np)
    cfg = _make_cfg()

    # base oracle + the two chosen adversarial personas
    persona_labels: List[str] = [BASE_LABEL, PERSONA_A, PERSONA_B]
    overlay_personas: List[str] = [PERSONA_A, PERSONA_B]

    def _run_single(label, acqf, seed):
        if label == "RANDOM_SEARCH":
            return random_search_regrets(seed, utility=utility, plant=plant,
                                         bounds_np=bounds_np, u_star=u_star)
        return run_single_seed(label, acqf, seed, utility=utility, plant=plant,
                               bounds=bounds, u_star=u_star, cfg=cfg)

    if INCLUDE_RANDOM_ANCHOR:
        persona_labels.append("RANDOM_SEARCH")
        overlay_personas.append("RANDOM_SEARCH")

    results = build_results(ACQF_TYPES, persona_labels, _run_single)

    # Save raw regret arrays for later reuse.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for acqf, per_persona in results.items():
        for label, arr in per_persona.items():
            np.save(OUT_DIR / f"regret_{acqf}_{label}.npy", arr)

    # One overlay figure per acqf (base + the two adverse personas).
    saved = save_per_acqf_figures(
        results, OUT_DIR, personas=overlay_personas,
        eps=EPS_CONV, u_star=u_star, show_seeds=False,
    )
    for p in saved:
        print(f"Figure saved → {p}")

    # Combined grid (handy when ACQF_TYPES has more than one entry).
    if len(ACQF_TYPES) > 1:
        grid = plot_acqf_grid(results, personas=overlay_personas,
                              eps=EPS_CONV, u_star=u_star, ncols=2)
        grid_path = OUT_DIR / "acqf_grid.png"
        grid.savefig(grid_path, dpi=150, bbox_inches="tight")
        print(f"Grid saved → {grid_path}")

    # Robustness-gap summary (final-iteration regret gap vs base; lower = better).
    print("\n=== Robustness gap (final-iter regret − base; lower is better) ===")
    for acqf, row in robustness_gap_table(results).items():
        pretty = "  ".join(f"{k}={v:+.3f}" for k, v in row.items())
        print(f"  {acqf:<7} {pretty}")


if __name__ == "__main__":
    main()
