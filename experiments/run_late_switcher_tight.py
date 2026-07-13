"""
run_late_switcher_tight.py
==========================
LATE-SWITCHER-ONLY variant of run_validator_comparison, with two changes
agreed after the first full run:

  1. MORE ITERATIONS: 40 instead of 20, switch at 12 instead of 8, so the
     post-switch phase (28 iters) is at least as long as a full pre-switch
     convergence.  Validator retest budget bumped accordingly (5 -> 8).

  2. TIGHTER SECOND TARGET: target 2 is tighter than target 1 on EVERY
     metric (lower overshoot, faster settling, lower MSE).  Moving to it only
     ever *decreases* metrics, so the directional constraints accumulated
     pre-switch (all "<=" vs incumbent) never bind — the preference switch is
     the only adversarial signal, cleanly isolating the validator's
     contribution.  (The seed-123/456 pair of the main experiment does not
     control the direction of target 2; here the pair was selected by scanning
     reachable targets: seed 95 -> seed 24, normalised distance 0.300.)

Story: the human first accepts a slow, bouncy response, then changes their
mind and demands a faster, better-damped one.

Standalone by design: nothing in the existing drivers is imported or
modified, so it can be prepared/queued while run_validator_comparison is
still running.  Results go to their own directory (late_switcher_tight).

Run:  python run_late_switcher_tight.py
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
    TargetSeekingUtility,
    generate_random_target,
    normalize_metrics,
    compute_ground_truth_optimum,
)
from dynamic_oracle import GenuineLateSwitcherOracle

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QUICK-CHANGE CONFIG                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ARMS = ["pbo", "l2g", "validated"]

ACQF_TYPE = "qEUBO"
ACQF_BETA = 2.0

# Extended budget: 28 post-switch iterations (>= one full convergence)
SEEDS        = [42, 7, 13, 99, 2025]
N_ITERATIONS = 40
SWITCH_ITER  = 12
WARM_START_N = 20
N_RETRIES    = 10

RETEST_DELAY = 1
VALIDATOR_KWARGS = dict(
    tau_flag=0.25,
    min_iter=5,
    retest_delay=RETEST_DELAY,
    max_retests_total=8,          # was 5; 40 iters need more retest budget
    max_retests_per_pair=1,
    confirm_weight=2,
    correction_weight=2,
    changepoint_confirmations=2,
    changepoint_window=5,
    post_change_weight=2,
    verbose=True,
)

# Plant + search space (identical to the main experiment)
PLANT_WN, PLANT_ZETA = 1.0, 0.7
BOUNDS_KP = (0.1, 5.0)
BOUNDS_KI = (0.001, 3.0)

# Loose -> tight target pair (selected by scanning reachable targets so that
# target 2 is tighter than target 1 on every normalised metric, d = 0.300)
TARGET1_SEED    = 95      # loose:  ov=48.55%, ts=12.77s, mse=0.0210
TARGET2_SEED    = 24      # tight:  ov=21.76%, ts= 7.37s, mse=0.0185
MIN_SWITCH_DIST = 0.20    # sanity floor on the normalised target distance
UTILITY_WEIGHTS = (1.0, 1.0, 1.0)
GRID_N          = 30

OUT_DIR = (Path(__file__).parent / "experiments" / "results"
           / "late_switcher_tight")

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
    """Loose target 1, strictly tighter target 2, and their optima."""
    t1, p1 = generate_random_target(plant, bounds_np, seed=TARGET1_SEED)
    u1 = TargetSeekingUtility(t1, weights=UTILITY_WEIGHTS)
    print(f"Target 1 (loose): ov={t1['overshoot_pct']:.2f}%, "
          f"ts={t1['settling_time']:.2f}s, mse={t1['tracking_mse']:.4f}  "
          f"(Kp={p1[0]:.3f}, Ki={p1[1]:.3f})")

    t2, p2 = generate_random_target(plant, bounds_np, seed=TARGET2_SEED)
    u2 = TargetSeekingUtility(t2, weights=UTILITY_WEIGHTS)
    print(f"Target 2 (tight): ov={t2['overshoot_pct']:.2f}%, "
          f"ts={t2['settling_time']:.2f}s, mse={t2['tracking_mse']:.4f}  "
          f"(Kp={p2[0]:.3f}, Ki={p2[1]:.3f})")

    # Guard the design assumptions: tighter on every metric, visible distance.
    n1, n2 = normalize_metrics(t1), normalize_metrics(t2)
    assert np.all(n2 <= n1 + 1e-9), (
        f"target 2 is NOT tighter on every metric: n1={n1}, n2={n2}")
    d = float(np.linalg.norm(n2 - n1))
    assert d >= MIN_SWITCH_DIST, f"targets too close (d={d:.3f})"
    print(f"Tightness check OK: target 2 <= target 1 on every metric, d={d:.3f}")

    print("Grid search for U*_1 (~30-90 s) ...")
    _bp1, u1_star, _g1 = compute_ground_truth_optimum(plant, bounds_np, u1, n_grid=GRID_N)
    print(f"U*_1 = {u1_star:.4f}")
    print("Grid search for U*_2 (~30-90 s) ...")
    _bp2, u2_star, _g2 = compute_ground_truth_optimum(plant, bounds_np, u2, n_grid=GRID_N)
    print(f"U*_2 = {u2_star:.4f}")
    return u1, u2, u1_star, u2_star


def main():
    from run_loop_validated import run_loop_validated  # lazy: torch/botorch

    plant, bounds, bounds_np = build_plant_and_bounds()
    u1, u2, u1_star, u2_star = build_utilities_and_gt(plant, bounds_np)
    cfg = {"experiment": {"n_iterations": N_ITERATIONS,
                          "warm_start_n": WARM_START_N,
                          "n_retries": N_RETRIES}}
    u_star_fn = lambda t: u1_star if t < SWITCH_ITER else u2_star

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    persona = "LATE_SWITCHER"
    results: Dict[str, Dict[str, np.ndarray]] = {persona: {}}
    for arm in ARMS:
        print(f"\n{'#'*64}\n#  persona={persona} (tight switch)  arm={arm}  "
              f"acqf={ACQF_TYPE}\n{'#'*64}")
        curves: List[np.ndarray] = []
        events_per_seed = []
        for seed in SEEDS:
            print(f"\n--- seed {seed} ---")
            oracle = GenuineLateSwitcherOracle(u1, u2, seed=seed, verbose=False,
                                               switch_iter=SWITCH_ITER)
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
        np.save(OUT_DIR / f"regret_{persona}_{arm}.npy", arr)
        with open(OUT_DIR / f"events_{persona}_{arm}.json", "w") as f:
            json.dump(events_per_seed, f, indent=2, default=str)

    # Figures (both band variants, via the shared plotting module)
    from plot_validator_comparison import plot_all
    meta = {"acqf": ACQF_TYPE, "switch_iter": SWITCH_ITER,
            "retest_delay": RETEST_DELAY, "n_seeds": len(SEEDS),
            "u1_star": u1_star, "u2_star": u2_star,
            "target1_seed": TARGET1_SEED, "target2_seed": TARGET2_SEED,
            "n_iterations": N_ITERATIONS}
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    for p in plot_all(results, OUT_DIR, meta):
        print(f"Figure saved → {p}")

    print("\n=== Final-iteration regret (mean over seeds) ===")
    row = "  ".join(f"{arm}={results[persona][arm][:, -1].mean():.4f}"
                    for arm in ARMS)
    print(f"  {persona:<16} {row}")


if __name__ == "__main__":
    main()
