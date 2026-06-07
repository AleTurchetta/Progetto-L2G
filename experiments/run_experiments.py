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

# Re-use all helpers from the single-seed script
from run_experiments_single import (
    load_config,
    build_oracle,
    build_plant_and_bounds,
    set_all_seeds,
    run_loop,
)
from oracle import compute_ground_truth_optimum

# ============================================================
# QUICK-CHANGE SECTION — edit these three lines to switch runs
# ============================================================
YAML_CONFIG_PATH = Path(__file__).parent / "configs" / "s1_monotone.yaml"   # ← swap to any scenario YAML
ACQF_TYPE        = "qUCB"               # "qUCB" | "qEI" | "qLogEI" | "qSR"
ACQF_BETA        = 3.0                  # UCB beta (ignored for qEI / qLogEI / qSR)
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
):
    """
    Two-panel figure:
      Left  — mean ± std simple regret curve across all seeds
      Right — oracle utility heatmap with the last seed's trajectory overlaid

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
    """
    persona_name = cfg["persona"]
    n_iters      = len(mean_regrets)
    n_seeds      = len(seeds)
    iters        = np.arange(1, n_iters + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"PBO  ·  {persona_name}  ·  {acqf_type}  ·  {n_seeds} seeds",
        fontsize=13,
    )

    # ── Mean ± std simple regret ─────────────────────────────────────────────
    # Plot individual seed traces faintly
    for i, row in enumerate(all_regrets):
        ax1.plot(iters, row, color="steelblue", alpha=0.15, linewidth=1)

    # Mean line + shaded ±1 std band
    ax1.plot(iters, mean_regrets, "b-o", linewidth=2.5, markersize=6, label="Mean regret")
    ax1.fill_between(
        iters,
        mean_regrets - std_regrets,
        mean_regrets + std_regrets,
        alpha=0.25,
        color="steelblue",
        label="±1 std",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Simple Regret  (U* − U_best)")
    ax1.set_title(f"Simple Regret  (mean ± std, {n_seeds} seeds)")
    ax1.set_xlim(0.5, n_iters + 0.5)
    ax1.legend(fontsize=9)

    # Annotate the first iteration where mean regret drops below 0.05
    CONVERGENCE_THRESHOLD = 0.05
    conv_iters = np.where(mean_regrets <= CONVERGENCE_THRESHOLD)[0]
    if len(conv_iters) > 0:
        ci = conv_iters[0] + 1  # 1-indexed
        ax1.axvline(ci, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
        ax1.text(
            ci + 0.1, ax1.get_ylim()[1] * 0.9,
            f"conv. @ iter {ci}",
            color="orange", fontsize=8,
        )

    # ── Utility heatmap (last seed trajectory) ───────────────────────────────
    kp_unique = np.unique(grid_data["Kp"])
    ki_unique = np.unique(grid_data["Ki"])
    U_2d = grid_data["utility"].reshape(len(kp_unique), len(ki_unique))

    cf = ax2.contourf(kp_unique, ki_unique, U_2d.T, levels=50, cmap="viridis")
    plt.colorbar(cf, ax=ax2, label="Oracle Utility")

    if best_traj_x_last:
        traj = np.array(best_traj_x_last)
        ax2.plot(traj[:, 0], traj[:, 1], "w-o", markersize=5, linewidth=1.5,
                 label=f"Trajectory (seed {seeds[-1]})", zorder=4)
        ax2.scatter(traj[0, 0], traj[0, 1], c="cyan", s=80, zorder=5,
                    label="Start", edgecolors="k", linewidths=0.5)
        ax2.scatter(traj[-1, 0], traj[-1, 1], c="red", s=100, zorder=5,
                    label="Final best", edgecolors="k", linewidths=0.5)

    best_gt_idx = int(np.argmax(grid_data["utility"]))
    ax2.scatter(
        grid_data["Kp"][best_gt_idx], grid_data["Ki"][best_gt_idx],
        marker="*", c="yellow", s=220, zorder=6, label="U* (ground truth)",
        edgecolors="k", linewidths=0.5,
    )

    ax2.set_xlabel("Kp")
    ax2.set_ylabel("Ki")
    ax2.set_title(f"Utility Landscape  (trajectory from seed {seeds[-1]})")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"multi_{n_seeds}seeds_{acqf_type}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {fname}")
    plt.show()


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
        best_traj_x_last = best_traj_x   # keep last seed for heatmap overlay

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
        cfg, ACQF_TYPE, seeds,
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
