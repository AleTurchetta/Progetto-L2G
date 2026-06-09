"""
compare_acqf_plots.py
=====================
Plotting helpers for the acquisition-function comparison.

These are PLOTTING FUNCTIONS ONLY — they take pre-computed regret arrays and
draw them.  The experiment script (written later) is responsible for running the
oracle-in-the-loop PBO and producing the `results` dict consumed here.

For each acquisition function tested, one panel overlaying the Simple-Regret
curves of the base (consistent) oracle and two adversarial personas

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    pass

from convergence_utils import convergence_iteration_mean, final_regret, DEFAULT_EPS

# ── Styling ───────────────────────────────────────────────────────────────────
BASE_LABEL = "CONSISTENT_BASE"
_BASE_COLOUR = "#222222"               # reference oracle drawn in bold dark grey
_PERSONA_COLOURS = [
    "#C44E52", "#DD8452", "#8172B3", "#55A868",
    "#4C72B0", "#937860", "#DA8BC3", "#8C8C8C",
]


def _curve_colour(label: str, persona_order: Sequence[str]) -> str:
    if label == BASE_LABEL:
        return _BASE_COLOUR
    others = [p for p in persona_order if p != BASE_LABEL]
    return _PERSONA_COLOURS[others.index(label) % len(_PERSONA_COLOURS)]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SINGLE PANEL — one acquisition function, base + personas overlaid
# ─────────────────────────────────────────────────────────────────────────────

def plot_acqf_overlay(
    persona_curves: Dict[str, np.ndarray],
    acqf_label: str,
    *,
    u_star: float = 0.0,
    eps: float = DEFAULT_EPS,
    ax: Optional[plt.Axes] = None,
    show_seeds: bool = False,
    annotate_convergence: bool = True,
    log_scale: bool = False,
    base_label: str = BASE_LABEL,
) -> plt.Axes:
    """
    Draw one acquisition function's regret curves (base + personas) on one axes.

    Parameters
    ----------
    persona_curves : {persona_label: ndarray [n_seeds, n_iters]}
        Must contain `base_label`; any other keys are treated as personas.
    acqf_label : str
        Used in the panel title.
    u_star : float
        Ground-truth optimum (≈ 0 with the target-seeking utility); drawn as the
        y=0 reference line label only.
    eps : float
        Utility-gap convergence threshold (for the optional convergence marker).
    show_seeds : bool
        Draw faint individual-seed traces.
    annotate_convergence : bool
        Mark the convergence iteration of each curve's MEAN with a vertical dotted line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))

    # Order: base first (drawn last so it sits on top), then personas as given.
    persona_order = [base_label] + [k for k in persona_curves if k != base_label]

    for label in persona_order:
        if label not in persona_curves:
            continue
        curves = np.atleast_2d(np.asarray(persona_curves[label], dtype=float))
        n_seeds = curves.shape[0]
        iters = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        colour = _curve_colour(label, persona_order)
        is_base = (label == base_label)

        if show_seeds and n_seeds > 1:
            for sc in curves:
                ax.plot(iters, sc, color=colour, alpha=0.10, linewidth=0.7)

        ax.plot(
            iters, mean, color=colour,
            linewidth=2.8 if is_base else 1.8,
            linestyle="-" if is_base else "--",
            marker="o", markersize=4,
            zorder=5 if is_base else 3,
            label=f"{label}{' (ref)' if is_base else ''}  (n={n_seeds})",
        )
        ax.fill_between(iters, mean - std, mean + std, color=colour,
                        alpha=0.20 if is_base else 0.12, zorder=2)

        if annotate_convergence:
            t_conv = convergence_iteration_mean(curves, eps=eps)
            if t_conv is not None:
                ax.axvline(t_conv, color=colour, linestyle=":", linewidth=1.0,
                           alpha=0.7, zorder=1)

    ax.axhline(0.0, color="green", linestyle="--", linewidth=1.0, alpha=0.5,
               label=f"U* (regret=0), U*={u_star:.4f}")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Simple Regret  $SR_t = U^* - U(x_t^{\mathrm{best}})$", fontsize=10)
    ax.set_ylim(bottom=0)
    if log_scale:
        ax.set_yscale("log")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_title(f"Acquisition: {acqf_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ONE FIGURE PER ACQUISITION FUNCTION  (the deliverable you described)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_acqf_figures(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    personas: Optional[Sequence[str]] = None,
    eps: float = DEFAULT_EPS,
    u_star: float = 0.0,
    show_seeds: bool = False,
    log_scale: bool = False,
    base_label: str = BASE_LABEL,
) -> List[Tuple[str, plt.Figure]]:
    """
    Build ONE standalone figure per acquisition function (base + chosen personas
    overlaid).  Returns a list of (acqf_label, Figure) — ready to be saved.

    Parameters
    ----------
    results : {acqf_label: {persona_label: ndarray [n_seeds, n_iters]}}
    personas : optional list of persona labels to overlay (besides the base).
        If None, every non-base persona present is shown.
    """
    figs: List[Tuple[str, plt.Figure]] = []
    for acqf_label, persona_curves in results.items():
        if personas is not None:
            subset = {base_label: persona_curves[base_label]}
            subset.update({p: persona_curves[p] for p in personas if p in persona_curves})
        else:
            subset = persona_curves
        fig, ax = plt.subplots(figsize=(7.5, 5))
        plot_acqf_overlay(
            subset, acqf_label, u_star=u_star, eps=eps, ax=ax,
            show_seeds=show_seeds, log_scale=log_scale, base_label=base_label,
        )
        fig.tight_layout()
        figs.append((acqf_label, fig))
    return figs


def plot_acqf_grid(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    personas: Optional[Sequence[str]] = None,
    ncols: int = 2,
    eps: float = DEFAULT_EPS,
    u_star: float = 0.0,
    show_seeds: bool = False,
    log_scale: bool = False,
    base_label: str = BASE_LABEL,
    suptitle: str = "Acquisition-function comparison — regret vs persona",
) -> plt.Figure:
    """
    Same content as `plot_per_acqf_figures` but tiled into a single grid figure
    (one panel per acqf) for a quick side-by-side overview.
    """
    acqfs = list(results.keys())
    n = len(acqfs)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows),
                             squeeze=False)

    for idx, acqf_label in enumerate(acqfs):
        ax = axes[idx // ncols][idx % ncols]
        persona_curves = results[acqf_label]
        if personas is not None:
            subset = {base_label: persona_curves[base_label]}
            subset.update({p: persona_curves[p] for p in personas if p in persona_curves})
        else:
            subset = persona_curves
        plot_acqf_overlay(
            subset, acqf_label, u_star=u_star, eps=eps, ax=ax,
            show_seeds=show_seeds, log_scale=log_scale, base_label=base_label,
        )

    # blank any unused cells
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save_per_acqf_figures(
    results: Dict[str, Dict[str, np.ndarray]],
    out_dir: str | Path,
    *,
    personas: Optional[Sequence[str]] = None,
    dpi: int = 150,
    prefix: str = "acqf",
    **kwargs,
) -> List[Path]:
    """Render one PNG per acqf into `out_dir`; returns the saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for acqf_label, fig in plot_per_acqf_figures(results, personas=personas, **kwargs):
        safe = acqf_label.replace(" ", "_").replace("/", "-")
        path = out_dir / f"{prefix}_{safe}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPTIONAL SUMMARY — robustness gap table / bars (handy when choosing)
# ─────────────────────────────────────────────────────────────────────────────

def robustness_gap_table(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    base_label: str = BASE_LABEL,
    at_iteration: int = -1,
) -> Dict[str, Dict[str, float]]:
    """
    For each acqf, the final-iteration robustness gap of each persona:
        gap = mean_final_regret(persona) - mean_final_regret(base)
    Lower (closer to 0) is better.  Returns {acqf: {persona: gap}}.
    """
    table: Dict[str, Dict[str, float]] = {}
    for acqf_label, persona_curves in results.items():
        base_curves = np.atleast_2d(persona_curves[base_label])
        col = at_iteration - 1 if at_iteration > 0 else -1
        base_mean = float(base_curves[:, col].mean())
        row: Dict[str, float] = {}
        for label, curves in persona_curves.items():
            if label == base_label:
                continue
            row[label] = float(np.atleast_2d(curves)[:, col].mean()) - base_mean
        table[acqf_label] = row
    return table
