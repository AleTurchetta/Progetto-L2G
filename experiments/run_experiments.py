"""
run_experiments.py
==================
Multi-seed oracle-in-the-loop PBO experiment.

Runs N seeds, collects simple regret at every iteration, and plots:
  • Mean ± std simple regret curve (left panel)
  • Oracle utility heatmap with the last seed's trajectory overlaid (right panel)

All core logic is imported from run_experiments_single.py — only the
QUICK-CHANGE section, the outer seed loop, and the averaged plots live here.

Quick-change section
--------------------
Edit only the three constants at the top to switch scenarios and AcqF:
    YAML_CONFIG_PATH  — which scenario YAML to load
    ACQF_TYPE         — acquisition function
    ACQF_BETA         — UCB exploration coefficient (ignored for EI/SR)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

# Re-use all helpers from the single-seed script
from run_experiments_single import (
    load_config,
    build_oracle,
    build_plant_and_bounds,
    set_all_seeds,
    run_loop,
)
from oracle import compute_ground_truth_optimum

# Analysis plotting functions (panel-level)
from plot_regret import plot_regret_multi as _plot_regret_multi_panel
from plot_heatmap import plot_heatmap_panel as _plot_heatmap_panel

# ============================================================
# QUICK-CHANGE SECTION — edit these three lines to switch runs
# ============================================================
YAML_CONFIG_PATH = Path(__file__).parent / "configs" / "s1_monotone.yaml"   # ← swap to any scenario YAML
ACQF_TYPE        = "qUCB"               # "qUCB" | "qEI" | "qLogEI" | "qSR"
ACQF_BETA        = 2.0                  # UCB beta (ignored for qEI / qLogEI / qSR)
# ============================================================


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — multi-seed averaged
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_multi(
    mean_regrets: np.ndarray,
    std_regrets:  np.ndarray,
    all_regrets:  np.ndarray,
    best_traj_x_last,
    grid_data:    dict,
    cfg:          dict,
    acqf_type:    str,
    seeds:        list,
    U_star:       float,
):
    """
    Two-panel figure assembled from analysis-folder panel functions:
      Left  — mean ± std simple regret across all seeds  (plot_regret.plot_regret_multi)
      Right — oracle utility heatmap with last seed's trajectory  (plot_heatmap.plot_heatmap_panel)

    Saves to cfg["logging"]["output_dir"] and closes the figure without
    blocking the terminal (plt.show() is intentionally omitted).

    Parameters
    ----------
    mean_regrets      : shape [n_iters]  — mean simple regret per iteration
    std_regrets       : shape [n_iters]  — std of simple regret per iteration
    all_regrets       : shape [n_seeds, n_iters]  — raw per-seed regrets
    best_traj_x_last  : list[(Kp, Ki)]  — best trajectory from the last seed
    grid_data         : dict from compute_ground_truth_optimum
    cfg               : loaded YAML config
    acqf_type         : acquisition function label
    seeds             : list of seeds used
    U_star            : ground-truth utility maximum
    """
    persona_name = cfg["persona"]
    n_seeds      = len(seeds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"PBO  ·  {persona_name}  ·  {acqf_type}  ·  {n_seeds} seeds",
        fontsize=13,
    )

    # Left panel — mean ± std regret with individual seed traces and convergence annotation
    _plot_regret_multi_panel(
        all_regrets,
        u_star=U_star,
        title=f"Simple Regret  (mean ± std, {n_seeds} seeds)",
        ax=ax1,
        show_seeds=True,
        annotate_convergence=True,
    )

    # Right panel — utility heatmap with last seed's best-so-far trajectory
    _plot_heatmap_panel(
        grid_data,
        best_traj_x_last,
        title=f"Utility Landscape  (trajectory from seed {seeds[-1]})",
        seed_label=f"seed {seeds[-1]}",
        ax=ax2,
    )

    plt.tight_layout()

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"multi_{n_seeds}seeds_{acqf_type}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {fname}")
    plt.close("all")   # avoids blocking the terminal (no plt.show())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg           = load_config(YAML_CONFIG_PATH)
    plant, bounds = build_plant_and_bounds(cfg)
    seeds         = cfg["experiment"]["seeds"]

    # ── Ground-truth optimum (computed once — utility() is deterministic) ────
    print("Computing ground truth optimum (this may take ~30–90 s) ...")
    oracle_gt = build_oracle(cfg, seed=0, verbose=False)
    oracle_gt.noise_level = 0.0
    best_params, U_star, grid_data = compute_ground_truth_optimum(
        plant, bounds.numpy(), oracle_gt, n_grid=30
    )
    print(f"U* = {U_star:.4f}  at  Kp*={best_params[0]:.3f}, Ki*={best_params[1]:.3f}")

    # ── Multi-seed loop ──────────────────────────────────────────────────────
    all_regrets       = []
    best_traj_x_last  = None

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  RUN {run_idx + 1}/{len(seeds)}  —  seed = {seed}")
        print(f"{'='*60}")

        # Fresh oracle per seed so noise / late-switch state is independent
        oracle = build_oracle(cfg, seed=seed, verbose=False)

        regrets, best_traj_x, _, _, _ = run_loop(
            cfg, oracle, plant, bounds, U_star, ACQF_TYPE, ACQF_BETA, seed
        )
        all_regrets.append(regrets)
        best_traj_x_last = best_traj_x

        print(
            f"  Seed {seed} complete — "
            f"final regret = {regrets[-1]:.4f}, "
            f"min regret = {min(regrets):.4f}"
        )

    # ── Aggregate ────────────────────────────────────────────────────────────
    all_regrets  = np.array(all_regrets)        # [n_seeds, n_iters]
    mean_regrets = all_regrets.mean(axis=0)
    std_regrets  = all_regrets.std(axis=0)

    # Save raw regret data for later analysis
    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"regrets_{ACQF_TYPE}.npy", all_regrets)
    print(f"\nRaw regret array saved → {out_dir / f'regrets_{ACQF_TYPE}.npy'}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results_multi(
        mean_regrets, std_regrets, all_regrets,
        best_traj_x_last, grid_data,
        cfg, ACQF_TYPE, seeds, U_star,
    )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Persona  : {cfg['persona']}")
    print(f"AcqF     : {ACQF_TYPE}  (beta={ACQF_BETA})")
    print(f"Seeds    : {seeds}")
    print(f"{'Iter':<6} {'Mean Regret':>12} {'Std':>10}")
    for i, (m, s) in enumerate(zip(mean_regrets, std_regrets), start=1):
        print(f"{i:<6} {m:>12.4f} {s:>10.4f}")


if __name__ == "__main__":
    main()
