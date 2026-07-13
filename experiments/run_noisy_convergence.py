"""
run_noisy_convergence.py
========================
Companion to run_final_comparison: the NOISY persona ONLY, extended to 50
iterations to demonstrate CONVERGENCE under a noisy human.

Motivated by the extension study: at 30 iters the validator already wins, but
a longer horizon shows the qualitative story — the validated arm keeps
descending toward the optimum while both baselines plateau.  This requires the
validator's retest budget to scale with the horizon (budget exhaustion at the
default 10 caused a tail blow-up on one seed; 18 fixes it, and 24 is identical,
so 18 is comfortably sufficient):

    50 iters, budget 18  ->  validated @20=0.10 @30=0.09 @40=0.08 @50=0.07
                             pbo / l2g plateau ~0.12-0.14 throughout.

Everything else matches run_final_comparison (target 95, warm 20, noise 0.40,
same 8 seeds, qEUBO).  Writes to its own directory; run_final_comparison's
run-3 results are left untouched.  The output figure (validator_NOISY*.png at
50 iters) is intended to REPLACE the 30-iter noisy panel in the final set.

Run:  python run_noisy_convergence.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from base_oracle import (
    Persona, TargetSeekingUtility, make_oracle,
    generate_random_target, compute_ground_truth_optimum,
)
import run_final_comparison as base   # reuse the frozen final config

# ── overrides for the convergence demonstration ──────────────────────────────
N_ITERATIONS = 50
RETEST_BUDGET = 18          # ~ n_iters/3 (run-3 used 10 for 30 iters)
SPOTCHECK_EVERY = 3         # slightly more proactive over the longer horizon

VALIDATOR_KWARGS = dict(base.VALIDATOR_KWARGS)
VALIDATOR_KWARGS["max_retests_total"] = RETEST_BUDGET
VALIDATOR_KWARGS["spotcheck_every"] = SPOTCHECK_EVERY

OUT_DIR = (Path(__file__).parent / "experiments" / "results"
           / "noisy_convergence_50")


def main():
    from run_loop_validated import run_loop_validated

    plant, bounds, bounds_np = base.build_plant_and_bounds()
    t1, _ = generate_random_target(plant, bounds_np, seed=base.TARGET1_SEED)
    u1 = TargetSeekingUtility(t1, weights=base.UTILITY_WEIGHTS)
    print(f"Target (seed {base.TARGET1_SEED}): ov={t1['overshoot_pct']:.2f}%, "
          f"ts={t1['settling_time']:.2f}s, mse={t1['tracking_mse']:.4f}")
    print("Grid search for U* ...")
    _bp, u_star, _g = compute_ground_truth_optimum(plant, bounds_np, u1, n_grid=base.GRID_N)
    print(f"U* = {u_star:.4f}")

    cfg = {"experiment": {"n_iterations": N_ITERATIONS,
                          "warm_start_n": base.WARM_START_N,
                          "n_retries": base.N_RETRIES}}
    u_star_fn = lambda t: u_star

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"NOISY": {}}
    for arm in base.ARMS:
        print(f"\n{'#'*64}\n#  NOISY  arm={arm}  (50 iters, budget {RETEST_BUDGET})\n{'#'*64}")
        curves: List[np.ndarray] = []
        events_per_seed = []
        for seed in base.SEEDS:
            print(f"\n--- seed {seed} ---")
            oracle = make_oracle(Persona.NOISY, u1, seed=seed, verbose=False,
                                 noise_level=base.NOISE_LEVEL)
            regrets, _bt, _tx, _tr, _bi, extras = run_loop_validated(
                cfg, oracle, plant, bounds, u_star_fn, seed,
                arm=arm, acqf_type=base.ACQF_TYPE, acqf_beta=base.ACQF_BETA,
                validator_kwargs=VALIDATOR_KWARGS, verbose=True,
            )
            curves.append(np.asarray(regrets, dtype=float))
            events_per_seed.append({"seed": seed,
                                    "validation_iters": extras["validation_iters"],
                                    "events": extras["events"],
                                    "summary": extras["summary"]})
        arr = np.vstack(curves)
        results["NOISY"][arm] = arr
        np.save(OUT_DIR / f"regret_NOISY_{arm}.npy", arr)
        with open(OUT_DIR / f"events_NOISY_{arm}.json", "w") as f:
            json.dump(events_per_seed, f, indent=2, default=str)

    from plot_validator_comparison import plot_all
    meta = {"acqf": base.ACQF_TYPE, "n_seeds": len(base.SEEDS),
            "noise_level": base.NOISE_LEVEL, "n_iterations": N_ITERATIONS,
            "u_star": u_star, "target1_seed": base.TARGET1_SEED,
            "retest_budget": RETEST_BUDGET}
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    for p in plot_all(results, OUT_DIR, meta):
        print(f"Figure saved → {p}")

    print("\n=== NOISY convergence (mean over seeds) ===")
    for arm in base.ARMS:
        m = results["NOISY"][arm].mean(axis=0)
        print(f"  {arm:10s} @20={m[15:20].mean():.3f} @30={m[25:30].mean():.3f} "
              f"@40={m[35:40].mean():.3f} @50={m[45:50].mean():.3f}")


if __name__ == "__main__":
    main()
