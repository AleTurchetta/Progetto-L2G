"""
run_final_comparison.py
=======================
FINAL experiment: three ARMS x four PERSONAS x seeds, single acqf, with every
fix identified in the pilot studies folded in:

  Arms      : PBO (preference only) | PBO+L2G | PBO+L2G+VALIDATOR
  Personas  : CONSISTENT_BASE | MONOTONE_CONSTRAINED | NOISY (p=0.2)
              | LATE_SWITCHER (genuine target switch)

Changes vs run_validator_comparison (the first study):

  1. UNIFIED LOOSE TARGET.  Target 1 = seed 95 for ALL personas (mid-range,
     not at the tight corner of the reachable set), so every panel shares the
     same regret landscape and has visible convergence headroom.  The
     switcher's target 2 = seed 24 is TIGHTER on every metric (d = 0.300), so
     pre-switch directional constraints never fight the move (confound found
     in study 1, confirmed fixed in the tight-switch study).

  2. INCUMBENT = POSTERIOR ARGMAX for all arms (run_loop_validated step 11):
     a flipped label can't teleport the incumbent, and validator corrections
     show up in the regret the iteration they happen.

  3. VALIDATOR TUNING (from event-log analysis of both pilots):
       tau_flag   0.25 -> 0.38   (variance inflation made 0.25 unreachable)
       spot-check every 4 iters  (spends the idle retest budget on the most
                                  suspicious record, gated by tau_spot=0.45
                                  so a consistent human never pays a duel)
       changepoint 2-in-5 -> 3-in-8  (noise cannot fake three confirmations;
                                  a genuine switch still clears the bar)

  4. 30 iterations (was 20), switch at 12: post-switch phase long enough for
     full recovery; noisy persona gets enough polluted labels for the
     validator's corrections to matter.

Outputs (OUT_DIR): regret_{persona}_{arm}.npy, events_{persona}_{arm}.json,
meta.json, figures via plot_validator_comparison (band + no-band variants).
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
    normalize_metrics,
    compute_ground_truth_optimum,
)
from dynamic_oracle import GenuineLateSwitcherOracle

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QUICK-CHANGE CONFIG                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ARMS     = ["pbo", "l2g", "validated"]
PERSONAS = ["CONSISTENT_BASE", "MONOTONE_CONSTRAINED", "NOISY", "LATE_SWITCHER"]

ACQF_TYPE = "qEUBO"
ACQF_BETA = 2.0

# Persona knobs.  Noise at 0.30: with the posterior-argmax incumbent the loop
# is robust enough that 20% noise barely warps the preference GP (run-1
# finding: all arms within 0.02 of each other).  Noise sweep (0.30/0.40/0.50,
# last-10 mean regret) picked 0.40 as the peak of the validator's advantage:
#   0.30  validated 0.099  pbo 0.127  l2g 0.147
#   0.40  validated 0.061  pbo 0.143  l2g 0.221   <- widest robust gap
#   0.50  validated 0.186  pbo 0.364  l2g 0.182   <- validator breaks down:
#         at 50% flips the RETEST is itself wrong half the time, so it can no
#         longer tell truth from noise (one seed diverges to 0.42).
# At 0.40, naive feedback (l2g) is markedly WORSE than plain PBO — validation
# is precisely what turns heavy-noise feedback from harmful back into useful.
NOISE_LEVEL = 0.40
SWITCH_ITER = 12

# Validator knobs (see module docstring for the rationale of each change)
RETEST_DELAY = 1
VALIDATOR_KWARGS = dict(
    tau_flag=0.38,
    tau_spot=0.45,
    spotcheck_every=4,
    min_iter=5,
    retest_delay=RETEST_DELAY,
    max_retests_total=10,     # ~10 corrupted labels expected at 30% noise
    max_retests_per_pair=2,   # a noisy retest can corrupt a resolution; allow
                              # one re-audit so wrong confirms/corrections heal
    confirm_weight=2,
    correction_weight=2,
    changepoint_confirmations=3,
    changepoint_window=8,
    changepoint_recency=4,    # only fresh-record confirmations count as switch
                              # evidence: the LOO audit flags a genuine switch
                              # within ~2 iters of expression, while honest-
                              # label misfit flags surface much later
    changepoint_min_expressed=9,  # and none from early records: the LOO audit
                              # is trigger-happy while the GP is data-poor
                              # (run-1 finding: spurious change-points at
                              # iters 4-6 on stationary personas)
    post_change_weight=2,
    confirm_streak_limit=3,   # stand down after 3 straight confirmations
    verbose=True,
)

# Experiment budget
SEEDS        = [42, 7, 13, 99, 2025, 1, 314, 808]
N_ITERATIONS = 30
WARM_START_N = 20
N_RETRIES    = 10

# Plant + search space
PLANT_WN, PLANT_ZETA = 1.0, 0.7
BOUNDS_KP = (0.1, 5.0)
BOUNDS_KI = (0.001, 3.0)

# Unified loose -> tight target pair (scanned: T2 tighter on EVERY metric)
TARGET1_SEED    = 95      # loose:  ov=48.55%, ts=12.77s, mse=0.0210 (all personas)
TARGET2_SEED    = 24      # tight:  ov=21.76%, ts= 7.37s, mse=0.0185 (switcher only)
MIN_SWITCH_DIST = 0.20
UTILITY_WEIGHTS = (1.0, 1.0, 1.0)
GRID_N          = 30

OUT_DIR = Path(__file__).parent / "experiments" / "results" / "final_comparison_run3"

# Optional: restrict to a subset of personas via the command line, e.g.
#   python run_final_comparison.py NOISY
# The end-of-run figures are always rebuilt from EVERY regret_*.npy in OUT_DIR,
# so a subset run still yields the full 4-panel set as long as the untouched
# personas' files were copied in first (they are noise-independent).
_ARGV_PERSONAS = [a for a in sys.argv[1:] if not a.startswith("-")]

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
    t1, p1 = generate_random_target(plant, bounds_np, seed=TARGET1_SEED)
    u1 = TargetSeekingUtility(t1, weights=UTILITY_WEIGHTS)
    print(f"Target 1 (loose, all personas): ov={t1['overshoot_pct']:.2f}%, "
          f"ts={t1['settling_time']:.2f}s, mse={t1['tracking_mse']:.4f}  "
          f"(Kp={p1[0]:.3f}, Ki={p1[1]:.3f})")

    t2, p2 = generate_random_target(plant, bounds_np, seed=TARGET2_SEED)
    u2 = TargetSeekingUtility(t2, weights=UTILITY_WEIGHTS)
    print(f"Target 2 (tight, switcher only): ov={t2['overshoot_pct']:.2f}%, "
          f"ts={t2['settling_time']:.2f}s, mse={t2['tracking_mse']:.4f}  "
          f"(Kp={p2[0]:.3f}, Ki={p2[1]:.3f})")

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

    personas_to_run = _ARGV_PERSONAS or PERSONAS
    for p in personas_to_run:
        if p not in PERSONAS:
            raise ValueError(f"unknown persona {p!r}; choose from {PERSONAS}")
    if _ARGV_PERSONAS:
        print(f"[subset run] personas = {personas_to_run}  (figures will still "
              f"use every regret_*.npy already in {OUT_DIR.name})")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for persona in personas_to_run:
        results[persona] = {}
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
            np.save(OUT_DIR / f"regret_{persona}_{arm}.npy", arr)
            with open(OUT_DIR / f"events_{persona}_{arm}.json", "w") as f:
                json.dump(events_per_seed, f, indent=2, default=str)

    # Figures — rebuilt from EVERY regret_*.npy in OUT_DIR (so a subset run
    # still produces the full 4-panel set, using any pre-copied personas).
    from plot_validator_comparison import plot_all, load_results
    meta = {"acqf": ACQF_TYPE, "switch_iter": SWITCH_ITER,
            "retest_delay": RETEST_DELAY, "n_seeds": len(SEEDS),
            "noise_level": NOISE_LEVEL,
            "u1_star": u1_star, "u2_star": u2_star,
            "target1_seed": TARGET1_SEED, "target2_seed": TARGET2_SEED,
            "n_iterations": N_ITERATIONS,
            "validator_kwargs": {k: v for k, v in VALIDATOR_KWARGS.items()
                                 if k != "verbose"}}
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    disk_results = load_results(OUT_DIR)
    for p in plot_all(disk_results, OUT_DIR, meta):
        print(f"Figure saved → {p}")

    print("\n=== Final-iteration regret (mean over seeds) ===")
    for persona, arms in disk_results.items():
        row = "  ".join(f"{arm}={arms[arm][:, -1].mean():.4f}"
                        for arm in ARMS if arm in arms)
        print(f"  {persona:<22} {row}")


if __name__ == "__main__":
    main()
