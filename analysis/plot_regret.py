"""
Plotting utilities for simple regret convergence curves.

Functions
---------
plot_regret(regret_curve, u_star, title, ax, label, colour, annotate_convergence)
    Single run: plot SR_t vs iteration.

plot_regret_multi(regret_curves, u_star, title, ax, label, colour, show_seeds,
                  annotate_convergence)
    Multiple seeds: plot mean ± std band.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")


def plot_regret(
    regret_curve: np.ndarray,
    u_star: float,
    title: str = "Simple Regret",
    ax: plt.Axes = None,
    label: str = None,
    colour: str = "#4C72B0",
    annotate_convergence: bool = False,
) -> plt.Figure:
    """
    Plot a single simple-regret curve SR_t = U(x*) - best_utility_so_far_t.

    Parameters
    ----------
    regret_curve : 1D array of length n_iterations
        SR values computed externally (in run_experiment.py).
    u_star : float
        Oracle's global optimum utility — shown as reference.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to draw on.  Creates a new figure if None.
    label : str, optional
        Legend label for the curve.
    colour : str
        Line colour.
    annotate_convergence : bool
        If True, mark the first iteration where regret drops to ≤10% of the
        initial regret with an orange vertical line and text annotation.

    Returns
    -------
    fig : plt.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    regret_curve = np.asarray(regret_curve)
    iterations   = np.arange(1, len(regret_curve) + 1)
    ax.plot(iterations, regret_curve, color=colour, linewidth=2.0,
            marker="o", markersize=5, label=label or "SR")
    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"U* = {u_star:.4f}" if u_star is not None else "U*")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, len(regret_curve) + 0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9)

    if annotate_convergence and len(regret_curve) > 0 and regret_curve[0] > 0:
        conv_threshold = regret_curve[0] * 0.10
        conv_iters     = np.where(regret_curve <= conv_threshold)[0]
        if len(conv_iters) > 0:
            ci = int(conv_iters[0]) + 1  # 1-indexed
            ax.axvline(ci, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(
                ci + 0.1, float(regret_curve.max()) * 0.9,
                f"conv.\n@ iter {ci}\n(10% of R₀)",
                color="orange", fontsize=8,
            )

    if standalone:
        plt.tight_layout()
    return fig


def plot_regret_multi(
    regret_curves: np.ndarray,
    u_star: float,
    title: str = "Simple Regret (multi-seed)",
    ax: plt.Axes = None,
    label: str = None,
    colour: str = "#4C72B0",
    show_seeds: bool = False,
    annotate_convergence: bool = False,
) -> plt.Figure:
    """
    Plot mean ± std band over multiple seeds.

    Parameters
    ----------
    regret_curves : 2D array of shape (n_seeds, n_iterations)
        One row per seed.
    u_star : float
        Oracle's global optimum utility.
    show_seeds : bool
        If True, draw individual seed traces as faint lines behind the mean.
    annotate_convergence : bool
        If True, mark the first iteration where the *mean* regret drops to
        ≤10% of the initial mean regret with an orange vertical line and text.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    regret_curves = np.atleast_2d(regret_curves)
    iterations    = np.arange(1, regret_curves.shape[1] + 1)
    mean = regret_curves.mean(axis=0)
    std  = regret_curves.std(axis=0)
    n    = regret_curves.shape[0]

    if show_seeds:
        for seed_curve in regret_curves:
            ax.plot(iterations, seed_curve, color=colour, alpha=0.15,
                    linewidth=0.8)

    ax.plot(iterations, mean, color=colour, linewidth=2.0, marker="o",
            markersize=5, label=f"{label or 'SR'}  (n={n})")
    ax.fill_between(iterations, mean - std, mean + std,
                    color=colour, alpha=0.2, label="±1 std")
    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"U* = {u_star:.4f}" if u_star is not None else "U*")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, regret_curves.shape[1] + 0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9)

    if annotate_convergence and len(mean) > 0 and mean[0] > 0:
        conv_threshold = mean[0] * 0.10
        conv_iters     = np.where(mean <= conv_threshold)[0]
        if len(conv_iters) > 0:
            ci = int(conv_iters[0]) + 1  # 1-indexed
            ax.axvline(ci, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(
                ci + 0.1, float(mean.max()) * 0.9,
                f"conv. @ iter {ci}\n(10% of R₀)",
                color="orange", fontsize=8,
            )

    if standalone:
        plt.tight_layout()
    return fig
