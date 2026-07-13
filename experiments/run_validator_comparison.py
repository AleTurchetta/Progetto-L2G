"""
run_validator_comparison.py
===========================
END-GOAL experiment driver: three ARMS x three PERSONAS x seeds, single acqf.

    Arms      : PBO (preference only) | PBO+L2G | PBO+L2G+VALIDATOR
    Personas  : CONSISTENT_BASE | NOISY (p=0.2) | LATE_SWITCHER (genuine —
                the hidden utility TARGET changes at SWITCH_ITER)
    Acqf      : qEUBO (natively preferential; matches the duel paradigm)

Regret is DYNAMIC: measured against the persona's currently active utility.
For the genuine late switcher U*_1 and U*_2 are each computed once by grid
search; base and noisy use U*_1 throughout.

Outputs (OUT_DIR):
    regret_{persona}_{arm}.npy      [n_seeds, n_iterations]
    events_{persona}_{arm}.json     validator event logs per seed
    figures via plot_validator_comparison
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from base_oracle import (
    Persona,
    TargetSeekingUtility,
    MonotoneConstrainedOracle,
    make_oracle,
    generate_random_target,
    compute_ground_truth_optimum,
)
from dynamic_oracle import GenuineLateSwitcherOracle, generate_distinct_target

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QUICK-CHANGE CONFIG                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ARMS      = ["pbo", "l2g", "validated"]
PERSONAS  = ["CONSISTENT_BASE", "MONOTONE_CONSTRAINED", "NOISY", "LATE_SWITCHER"]

ACQF_TYPE = "qEUBO"
ACQF_BETA = 2.0

# Persona knobs
NOISE_LEVEL = 0.20
SWITCH_ITER = 8            # genuine utility-target switch (of N_ITERATIONS)

# Validator knobs — RETEST_DELAY is THE ablation parameter:
#   1 = instant (earliest possible; the flag itself needs one refit)
#   2-3 = human test-retest spacing (see ablation later)
RETEST_DELAY = 1
VALIDATOR_KWARGS = dict(
    tau_flag=0.25,
    min_iter=5,
    retest_delay=RETEST_DELAY,
    max_retests_total=5,
    max_retests_per_pair=1,
    confirm_weight=2,
    correction_weight=2,
    changepoint_confirmations=2,
    changepoint_window=5,
    post_change_weight=2,
    verbose=True,
)

# Experiment budget
SEEDS        = [42, 7, 13, 99, 2025]
N_ITERATIONS = 20
WARM_START_N = 20
N_RETRIES    = 10

# Plant + search space (matches earlier drivers)
PLANT_WN, PLANT_ZETA = 1.0, 0.7
BOUNDS_KP = (0.1, 5.0)
BOUNDS_KI = (0.001, 3.0)

# Targets
TARGET_SEED      = 123     # target 1 (shared with earlier experiments)
TARGET2_SEED     = 456     # search start for target 2
TARGET2_MIN_DIST = 0.25    # min normalised-metric distance between targets
UTILITY_WEIGHTS  = (1.0, 1.0, 1.0)
GRID_N           = 30

OUT_DIR = Path(__file__).parent / "experiments" / "results" / "validator_compare"

# ╚══════════════════════════════════════════════════════════════════════════╝


def build_plant_and_bounds():
    import torch
    import control_utils as cu
    plant = cu.make_plant(wn=PLANT_WN, zeta=PLANT_ZETA)
    bounds = torch.tensor(
        [[BOUNDS_KP[0], BOUNDS_KI[0]], [BOUNDS_KP[1], BOUNDS_KI[1]]],
        dtype=torch.double,
    )
    return plant, bounds, bounds.numpy()


def build_utilities_and_gt(plant, bounds_np):
    """Two reachable targets + their ground-truth optima."""
    t1, p1 = generate_random_target(plant, bounds_np, seed=TARGET_SEED)
    u1 = TargetSeekingUtility(t1, weights=UTILITY_WEIGHTS)
    print(f"Target 1: ov={t1['overshoot_pct']:.2f}%, ts={t1['settling_time']:.2f}s, "
          f"mse={t1['tracking_mse']:.4f}  (Kp={p1[0]:.3f}, Ki={p1[1]:.3f})")

    t2, p2 = generate_distinct_target(plant, bounds_np, t1,
                                      seed_start=TARGET2_SEED,
                                      min_dist=TARGET2_MIN_DIST)
    u2 = TargetSeekingUtility(t2, weights=UTILITY_WEIGHTS)
    print(f"Target 2: ov={t2['overshoot_pct']:.2f}%, ts={t2['settling_time']:.2f}s, "
          f"mse={t2['tracking_mse']:.4f}  (Kp={p2[0]:.3f}, Ki={p2[1]:.3f})")

    print("Grid search for U*_1 (~30-90 s) ...")
    _bp1, u1_star, _g1 = compute_ground_truth_optimum(plant, bounds_np, u1, n_grid=GRID_N)
    print(f"U*_1 = {u1_star:.4f}")
    print("Grid search for U*_2 (~30-90 s) ...")
    _bp2, u2_star, _g2 = compute_ground_truth_optimum(plant, bounds_np, u2, n_grid=GRID_N)
    print(f"U*_2 = {u2_star:.4f}")
    return u1, u2, u1_star, u2_star


def make_persona_oracle(label: str, u1, u2, seed: int):
    if label == "CONSISTENT_BASE":
        return make_oracle(Persona.CONSISTENT_BASE, u1, seed=seed, verbose=False)
    if label == "MONOTONE_CONSTRAINED":
        return MonotoneConstrainedOracle(u1, seed=seed, verbose=False)
    if label == "NOISY":
        return make_oracle(Persona.NOISY, u1, seed=seed, verbose=False,
                           noise_level=NOISE_LEVEL)
    if label == "LATE_SWITCHER":
        return GenuineLateSwitcherOracle(u1, u2, seed=seed, verbose=False,
                                         switch_iter=SWITCH_ITER)
    raise ValueError(label)


def make_u_star_fn(label: str, u1_star: float, u2_star: float):
    if label == "LATE_SWITCHER":
        return lambda t: u1_star if t < SWITCH_ITER else u2_star
    return lambda t: u1_star


def main():
    from run_loop_validated import run_loop_validated  # lazy: torch/botorch

    plant, bounds, bounds_np = build_plant_and_bounds()
    u1, u2, u1_star, u2_star = build_utilities_and_gt(plant, bounds_np)
    cfg = {"experiment": {"n_iterations": N_ITERATIONS,
                          "warm_start_n": WARM_START_N,
                          "n_retries": N_RETRIES}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    all_events: Dict[str, Dict[str, list]] = {}

    for persona in PERSONAS:
        results[persona] = {}
        all_events[persona] = {}
        u_star_fn = make_u_star_fn(persona, u1_star, u2_star)
        for arm in ARMS:
            print(f"\n{'#'*64}\n#  persona={persona}  arm={arm}  acqf={ACQF_TYPE}\n{'#'*64}")
            curves: List[np.ndarray] = []
            events_per_seed = []
            for seed in SEEDS:
                print(f"\n--- seed {seed} ---")
                oracle = make_persona_oracle(persona, u1, u2, seed)
                regrets, _bt, _tx, _tr, _bi, extras = run_loop_validated(
                    cfg, oracle, plant, bounds, u_star_fn, seed,
                    arm=arm, acqf_type=ACQF_TYPE, acqf_beta=ACQF_BETA,
                    validator_kwargs=VALIDATOR_KWARGS, verbose=True,
                )
                curves.append(np.asarray(regrets, dtype=float))
                events_per_seed.append({
                    "seed": seed,
                    "validation_iters": extras["validation_iters"],
                    "events": extras["events"],
                    "summary": extras["summary"],
                })
            arr = np.vstack(curves)
            results[persona][arm] = arr
            all_events[persona][arm] = events_per_seed
            np.save(OUT_DIR / f"regret_{persona}_{arm}.npy", arr)
            with open(OUT_DIR / f"events_{persona}_{arm}.json", "w") as f:
                json.dump(events_per_seed, f, indent=2, default=str)

    # Figures
    from plot_validator_comparison import plot_all
    meta = {"acqf": ACQF_TYPE, "switch_iter": SWITCH_ITER,
            "retest_delay": RETEST_DELAY, "n_seeds": len(SEEDS),
            "u1_star": u1_star, "u2_star": u2_star}
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    for p in plot_all(results, OUT_DIR, meta):
        print(f"Figure saved → {p}")

    # Console summary: final-iteration regret per (persona, arm)
    print("\n=== Final-iteration regret (mean over seeds) ===")
    for persona in PERSONAS:
        row = "  ".join(f"{arm}={results[persona][arm][:, -1].mean():.4f}"
                        for arm in ARMS)
        print(f"  {persona:<16} {row}")


if __name__ == "__main__":
    main()
