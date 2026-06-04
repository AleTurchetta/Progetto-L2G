"""
Plotting utilities for simple regret convergence curves.

Functions
---------
plot_regret(regret_curve, title)
    Single run: plot SR_t vs iteration.

plot_regret_multi(regret_curves, title)
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

    Returns
    -------
    fig : plt.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    iterations = np.arange(1, len(regret_curve) + 1)
    ax.plot(iterations, regret_curve, color=colour, linewidth=2.0,
            label=label or "SR")
    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"U* = {u_star:.4f}")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9)

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
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    iterations = np.arange(1, regret_curves.shape[1] + 1)
    mean = regret_curves.mean(axis=0)
    std  = regret_curves.std(axis=0)
    n    = regret_curves.shape[0]

    if show_seeds:
        for seed_curve in regret_curves:
            ax.plot(iterations, seed_curve, color=colour, alpha=0.15,
                    linewidth=0.8)

    ax.plot(iterations, mean, color=colour, linewidth=2.0,
            label=f"{label or 'SR'}  (n={n})")
    ax.fill_between(iterations, mean - std, mean + std,
                    color=colour, alpha=0.2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"U* = {u_star:.4f}")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
    return fig
