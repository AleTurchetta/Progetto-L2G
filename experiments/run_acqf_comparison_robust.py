"""
run_acqf_comparison_robust.py
=============================
ROBUST-loop acquisition-function comparison (separate from run_acqf_comparison.py).

Identical experiment to run_acqf_comparison.py — base oracle + two adversarial
personas, overlaid per acqf — but every run goes through run_loop_robust, which
applies the robustness fixes (trust-but-verify comparison cleaning, contradiction
de-dup, noise-tolerant PairwiseGP, robust incumbent selection).

Use this to see whether the fixes pull NOISY / LATE_SWITCHER toward the base
curve.  Keep run_acqf_comparison.py around as the "before" (baseline loop) and
diff the resulting figures.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

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
# ║  QUICK-CHANGE CONFIG                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Two ADVERSARIAL personas to overlay against the base oracle.
PERSONA_A = "NOISY"
PERSONA_B = "LATE_SWITCHER"

# All six acquisition functions (one overlay figure each + a combined grid).
ACQF_TYPES = ["qUCB", "qSR", "qEI", "qLogEI", "qEUBO", "qTS"]
ACQF_BETA  = 2.0

# Robustness knobs (run_loop_robust).
GROUNDING   = "pareto"   # "utility" (oracle-informed ceiling) | "pareto" (deployable) | "off"
PREF_JITTER = 1e-2        # PairwiseGP noise tolerance
DEDUP       = True        # resolve contradictory repeated comparisons

# Experiment budget.
SEEDS         = [42,  7, 13, 99, 2025]
N_ITERATIONS  = 20
WARM_START_N  = 20
N_RETRIES     = 10

# Plant + search space.
PLANT_WN, PLANT_ZETA = 1.0, 0.7
BOUNDS_KP = (0.1, 5.0)
BOUNDS_KI = (0.001, 3.0)

# Standardized target-seeking utility (shared by all personas / seeds).
TARGET_SEED     = 123
UTILITY_WEIGHTS = (1.0, 1.0, 1.0)
GRID_N          = 30
EPS_CONV        = DEFAULT_EPS

# Optional model-free absolute anchor.
INCLUDE_RANDOM_ANCHOR = False

# Output (separate from the baseline driver's directory).
OUT_DIR = Path(__file__).parent / "experiments" / "results" / f"acqf_compare_robust_{GROUNDING}"

PERSONA_KWARGS: Dict[str, dict] = {
    "NOISY":         {"noise_level": 0.20},
    "LATE_SWITCHER": {"switch_iter": 4},
}

# ╚══════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────────────
# Setup (lazy heavy imports)
# ─────────────────────────────────────────────────────────────────────────────

def build_plant_and_bounds():
    import torch
    import control_utils as cu
    plant = cu.make_plant(wn=PLANT_WN, zeta=PLANT_ZETA)
    bounds = torch.tensor(
        [[BOUNDS_KP[0], BOUNDS_KI[0]], [BOUNDS_KP[1], BOUNDS_KI[1]]],
        dtype=torch.double,
    )
    return plant, bounds, bounds.numpy()


def build_shared_utility_and_gt(plant, bounds_np):
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
    return {"experiment": {"n_iterations": N_ITERATIONS,
                           "warm_start_n": WARM_START_N,
                           "n_retries": N_RETRIES}}


# ─────────────────────────────────────────────────────────────────────────────
# Single robust run + model-free anchor
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(persona_label, acqf, seed, *, utility, plant, bounds, u_star, cfg):
    """One robust PBO run for (persona, acqf, seed) via run_loop_robust."""
    from run_loop_robust import run_loop_robust  # lazy: pulls torch/botorch
    kwargs = PERSONA_KWARGS.get(persona_label, {})
    oracle = make_oracle(Persona(persona_label), utility, seed=seed,
                         verbose=False, **kwargs)
    regrets, _bt, _tx, _tr, _bi = run_loop_robust(
        cfg, oracle, plant, bounds, u_star, acqf, ACQF_BETA, seed,
        grounding=GROUNDING, pref_jitter=PREF_JITTER, dedup=DEDUP,
    )
    return regrets


def random_search_regrets(seed, *, utility, plant, bounds_np, u_star):
    import base_oracle as bo
    rng = np.random.default_rng(seed)
    lo, hi = bounds_np[0], bounds_np[1]

    def u_of(kp, ki):
        return utility.utility(bo._eval_metrics(float(kp), float(ki), plant))

    best_u = -np.inf
    for _ in range(WARM_START_N):
        kp, ki = rng.uniform(lo, hi)
        best_u = max(best_u, u_of(kp, ki))
    regrets = []
    for _ in range(N_ITERATIONS):
        kp, ki = rng.uniform(lo, hi)
        best_u = max(best_u, u_of(kp, ki))
        regrets.append(u_star - best_u)
    return regrets


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration (pure — injectable run_single)
# ─────────────────────────────────────────────────────────────────────────────

def build_results(
    acqf_types: Sequence[str],
    persona_labels: Sequence[str],
    run_single: Callable[[str, str, int], Sequence[float]],
    seeds: Sequence[int] = tuple(SEEDS),
) -> Dict[str, Dict[str, np.ndarray]]:
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for acqf in acqf_types:
        print(f"\n{'#'*64}\n#  [ROBUST] ACQF = {acqf}\n{'#'*64}")
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

    print(f"\n[robust driver]  grounding={GROUNDING}  jitter={PREF_JITTER}  dedup={DEDUP}")
    results = build_results(ACQF_TYPES, persona_labels, _run_single)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for acqf, per_persona in results.items():
        for label, arr in per_persona.items():
            np.save(OUT_DIR / f"regret_{acqf}_{label}.npy", arr)

    saved = save_per_acqf_figures(
        results, OUT_DIR, personas=overlay_personas,
        eps=EPS_CONV, u_star=u_star, show_seeds=False, prefix="acqf_robust",
    )
    for p in saved:
        print(f"Figure saved → {p}")

    if len(ACQF_TYPES) > 1:
        grid = plot_acqf_grid(
            results, personas=overlay_personas, eps=EPS_CONV, u_star=u_star, ncols=2,
            suptitle=f"ROBUST loop ({GROUNDING}) — regret vs persona",
        )
        grid_path = OUT_DIR / "acqf_grid_robust.png"
        grid.savefig(grid_path, dpi=150, bbox_inches="tight")
        print(f"Grid saved → {grid_path}")

    print("\n=== Robustness gap (final-iter regret − base; lower is better) ===")
    for acqf, row in robustness_gap_table(results).items():
        pretty = "  ".join(f"{k}={v:+.3f}" for k, v in row.items())
        print(f"  {acqf:<7} {pretty}")


if __name__ == "__main__":
    main()
