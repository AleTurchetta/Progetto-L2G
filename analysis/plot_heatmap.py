"""
Plotting utilities for the ground-truth utility landscape over (Kp, Ki).

The grid arrays (Kp2d, Ki2d, U2d) come from oracle.compute_ground_truth_optimum(),
which you call once at the start of the experiment.

Functions
---------
plot_utility_heatmap(Kp2d, Ki2d, U2d, u_star, kp_best, ki_best)
    Filled contour map of U(Kp, Ki) with the optimum marked.

plot_utility_heatmap_with_trajectory(... , trajectory_kp, trajectory_ki)
    Same heatmap with BO candidate positions overlaid and coloured by iteration.

plot_heatmap_panel(grid_data, best_traj_x, title, seed_label, ax)
    Convenience wrapper used by run_experiments_single.py and run_experiments.py.
    Accepts the flat grid_data dict returned by compute_ground_truth_optimum and
    a list of (Kp, Ki) tuples for the best-trajectory overlay.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")

# Peak colour is dark red; low utility fades to pale yellow.
_CMAP_UTILITY = "YlOrRd"
_TRAJ_CMAP    = mpl.colormaps.get_cmap("coolwarm")   # early=blue, late=red


def plot_utility_heatmap(
    Kp2d: np.ndarray,
    Ki2d: np.ndarray,
    U2d:  np.ndarray,
    u_star:  float,
    kp_best: float,
    ki_best: float,
    title: str = "Ground-Truth Utility Landscape",
    n_contours: int = 10,
    vmin_percentile: int = 20,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Filled contour map of U(Kp, Ki).

    Parameters
    ----------
    Kp2d, Ki2d : 2D arrays
        Meshgrid arrays from np.meshgrid(kp_vals, ki_vals, indexing='ij').
        Shape (n_kp, n_ki).
    U2d : 2D array, same shape
        Oracle utility values on the grid.
    u_star : float
        Global optimum utility value (for colorbar reference label).
    kp_best, ki_best : float
        Coordinates of the global optimum (starred marker).
    n_contours : int
        Number of iso-utility contour lines overlaid on the filled map.
    vmin_percentile : int
        Percentile of U2d used as the colour floor (bottom vmin_percentile %
        is clipped to the lowest colour).  Default 20.

    Returns
    -------
    fig : plt.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    u_flat   = U2d.ravel()
    vmin_val = float(np.percentile(u_flat, vmin_percentile))
    vmax_val = float(u_flat.max())

    # Axes convention: x = Ki, y = Kp  (Ki varies faster in the search space)
    pcm = ax.pcolormesh(Ki2d, Kp2d, U2d,
                        cmap=_CMAP_UTILITY, shading="auto",
                        vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(pcm, ax=ax, label=f"Utility  U(Kp, Ki)  [U* = {u_star:.4f}]",
                 pad=0.02)

    ax.contour(Ki2d, Kp2d, U2d, levels=n_contours,
               colors="white", alpha=0.25, linewidths=0.6)

    # Global optimum marker
    ax.scatter([ki_best], [kp_best],
               marker="*", s=300, c="gold", edgecolors="black",
               linewidths=0.8, zorder=10,
               label=f"x*  (Kp={kp_best:.3f}, Ki={ki_best:.3f})")

    ax.set_xlabel("$K_i$", fontsize=12)
    ax.set_ylabel("$K_p$", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    if standalone:
        plt.tight_layout()
    return fig


def plot_utility_heatmap_with_trajectory(
    Kp2d: np.ndarray,
    Ki2d: np.ndarray,
    U2d:  np.ndarray,
    u_star:  float,
    kp_best: float,
    ki_best: float,
    trajectory_kp: np.ndarray,
    trajectory_ki: np.ndarray,
    title: str = "Utility Landscape with BO Trajectory",
    n_contours: int = 10,
    vmin_percentile: int = 20,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Heatmap with BO candidate positions overlaid, colour-coded by iteration.

    Parameters
    ----------
    trajectory_kp, trajectory_ki : 1D arrays of length n_iterations
        Kp and Ki values of the B candidate proposed at each BO iteration.
        Produced in run_experiment.py as the loop runs.

    All other parameters are the same as plot_utility_heatmap.
    """
    # Draw the base heatmap first
    fig = plot_utility_heatmap(
        Kp2d, Ki2d, U2d, u_star, kp_best, ki_best,
        title=title, n_contours=n_contours,
        vmin_percentile=vmin_percentile, ax=ax,
    )
    ax = fig.axes[0]

    n = len(trajectory_kp)
    if n == 0:
        return fig

    # Colour each candidate by iteration (blue=early, red=late)
    norm    = mpl.colors.Normalize(vmin=1, vmax=max(n, 2))
    colours = [_TRAJ_CMAP(norm(t)) for t in range(1, n + 1)]

    # Connecting line (faint white so it doesn't dominate)
    ax.plot(trajectory_ki, trajectory_kp,
            color="white", alpha=0.3, linewidth=1.0, zorder=5)

    # Scatter candidates
    ax.scatter(trajectory_ki, trajectory_kp,
               c=colours, s=60,
               edgecolors="white", linewidths=0.5, zorder=6)

    # Iteration colorbar
    sm = mpl.cm.ScalarMappable(cmap=_TRAJ_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.14, shrink=0.6, label="BO Iteration")

    if ax is None:
        plt.tight_layout()
    return fig


def plot_heatmap_panel(
    grid_data: dict,
    best_traj_x,
    title: str = "Utility Landscape + Trajectory",
    seed_label: str = "",
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Convenience panel function used directly by run_experiments_single.py and
    run_experiments.py.

    Accepts the flat ``grid_data`` dict returned by
    ``oracle.compute_ground_truth_optimum`` (keys: 'Kp', 'Ki', 'utility' — all
    1D flat arrays of length n_grid²) and a list of (Kp, Ki) tuples that record
    the best-so-far position after each iteration.

    The colormap floor is set to the 20th-percentile of the utility
    distribution so that gradient is visible across the high-utility region
    regardless of how far below the optimum the warm-start trajectory sits.

    Parameters
    ----------
    grid_data    : dict with keys 'Kp', 'Ki', 'utility' (1D flat arrays)
    best_traj_x  : list of (Kp, Ki) tuples — best position after each iteration
    title        : axes title
    seed_label   : appended to the trajectory legend entry if non-empty
    ax           : existing Axes to draw on; creates a new figure if None

    Returns
    -------
    fig : plt.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    kp_unique = np.unique(grid_data["Kp"])
    ki_unique = np.unique(grid_data["Ki"])
    U_2d      = grid_data["utility"].reshape(len(kp_unique), len(ki_unique))

    U_max     = float(grid_data["utility"].max())
    vmin_clip = float(np.percentile(grid_data["utility"], 20))
    levels    = np.linspace(vmin_clip, U_max, 50)

    cf = ax.contourf(kp_unique, ki_unique, U_2d.T, levels=levels,
                     cmap=_CMAP_UTILITY, extend="min")
    plt.colorbar(cf, ax=ax, label="Oracle Utility")

    # Best-so-far trajectory
    if best_traj_x:
        traj      = np.array(best_traj_x)
        traj_label = f"Best trajectory"
        if seed_label:
            traj_label += f" ({seed_label})"
        ax.plot(traj[:, 0], traj[:, 1], "w-o", markersize=5, linewidth=1.5,
                label=traj_label, zorder=4)
        ax.scatter(traj[0, 0], traj[0, 1], c="cyan", s=80, zorder=5,
                   label="Start", edgecolors="k", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c="red", s=100, zorder=5,
                   label="Final best", edgecolors="k", linewidths=0.5)

    # Ground-truth optimum star
    best_gt_idx = int(np.argmax(grid_data["utility"]))
    ax.scatter(
        grid_data["Kp"][best_gt_idx], grid_data["Ki"][best_gt_idx],
        marker="*", c="gold", s=220, zorder=6, label="U* (ground truth)",
        edgecolors="k", linewidths=0.5,
    )

    ax.set_xlabel("Kp")
    ax.set_ylabel("Ki")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")

    if standalone:
        plt.tight_layout()
    return fig
