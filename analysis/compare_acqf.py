"""
Side-by-side regret comparison for different acquisition function variants.

Functions
---------
plot_regret_comparison(regret_dict, u_star)
    One axes per variant, convergence curves side by side.
    regret_dict: {label: np.ndarray of shape (n_seeds, n_iterations)}

plot_final_regret_bars(regret_dict, u_star)
    Bar chart of final-iteration simple regret (mean ± std) per variant.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")

_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot_regret_comparison(
    regret_dict: dict,
    u_star: float,
    title: str = "Acquisition Function Comparison",
    show_seeds: bool = False,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot all variants on the same axes: mean ± std band, one colour per variant.

    Parameters
    ----------
    regret_dict : dict
        {label: np.ndarray of shape (n_seeds, n_iterations)}
        Each entry is the simple-regret matrix for one variant.
        SR_t = U* - best_utility_so_far_t, computed in run_experiment.py.
    u_star : float
        Oracle global optimum — used only for the y=0 reference line label.
    show_seeds : bool
        If True, draw individual seed traces as faint lines.
    log_scale : bool
        Log y-axis (useful when curves span orders of magnitude).

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, curves) in enumerate(regret_dict.items()):
        colour = _COLOURS[i % len(_COLOURS)]
        curves = np.atleast_2d(curves)
        n      = curves.shape[0]
        iters  = np.arange(1, curves.shape[1] + 1)
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)

        if show_seeds:
            for sc in curves:
                ax.plot(iters, sc, color=colour, alpha=0.12, linewidth=0.7)

        ax.plot(iters, mean, color=colour, linewidth=2.0,
                label=f"{label}  (n={n})")
        ax.fill_between(iters, mean - std, mean + std, color=colour, alpha=0.18)

    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"U* = {u_star:.4f}")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    if log_scale:
        ax.set_yscale("log")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    plt.tight_layout()
    return fig


def plot_final_regret_bars(
    regret_dict: dict,
    u_star: float,
    title: str = "Final-Iteration Regret",
    at_iteration: int = -1,
) -> plt.Figure:
    """
    Bar chart of simple regret at a fixed iteration (default: last).
    Error bars show ± 1 std across seeds.

    Parameters
    ----------
    regret_dict : dict
        Same format as plot_regret_comparison.
    at_iteration : int
        Which iteration to read.  -1 = last, 1-indexed otherwise.
    """
    labels = list(regret_dict.keys())
    means, stds = [], []

    for label in labels:
        curves = np.atleast_2d(regret_dict[label])
        col    = at_iteration - 1 if at_iteration > 0 else -1
        vals   = curves[:, col]
        means.append(vals.mean())
        stds.append(vals.std())

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 4))

    bars = ax.bar(x, means, yerr=stds, color=_COLOURS[:len(labels)],
                  alpha=0.82, width=0.5,
                  error_kw=dict(ecolor="black", capsize=5, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Simple Regret", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.5,
               label=f"U* = {u_star:.4f}")
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig