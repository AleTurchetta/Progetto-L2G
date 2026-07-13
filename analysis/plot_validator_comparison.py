"""
plot_validator_comparison.py
============================
Presentation-quality figures for the validator study: one panel per PERSONA,
three curves per panel (PBO / PBO+L2G / PBO+L2G+Validator), mean over seeds.

Every figure is exported twice: mean-only, and with an interquartile band
across seeds (`_band` suffix).  With four personas the combined figure is a
2x2 grid (slide-friendly); otherwise a 1xN row.

Palette (validated for colour-vision deficiency and 3:1 contrast on white;
identity is additionally carried by linestyle + marker, never colour alone):
    baseline gray #6e6e6e / L2G red #e34948 / Validator blue #2a78d6.

Usable two ways:
    • called by an experiment driver with results in memory;
    • standalone:  python plot_validator_comparison.py <results_dir>
      (re-plots from the saved regret_{persona}_{arm}.npy files).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#b0b0b0",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#e4e4e4",
    "grid.linewidth":    0.7,
    "axes.axisbelow":    True,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "axes.labelcolor":   "#222222",
    "text.color":        "#222222",
})

ARM_ORDER = ("pbo", "l2g", "validated")
ARM_LABELS = {
    "pbo":       "Standard PBO",
    "l2g":       "PBO + L2G",
    "validated": "PBO + L2G + Validator",
}
# Baseline recessive; hero series (validator) solid and heaviest.
ARM_STYLES = {
    "pbo":       dict(color="#6e6e6e", ls=":",  lw=1.8, marker="o", ms=4.5),
    "l2g":       dict(color="#e34948", ls="--", lw=2.0, marker="s", ms=4.0),
    "validated": dict(color="#2a78d6", ls="-",  lw=2.4, marker="D", ms=4.0),
}
PERSONA_TITLES = {
    "CONSISTENT_BASE":      "Consistent Human",
    "MONOTONE_CONSTRAINED": "Consistent Human, Constrained Feedback",
    "NOISY":                "Noisy Human",
    "LATE_SWITCHER":        "Late-Switching Human",
}
PERSONA_DESCRIPTIONS = {
    "CONSISTENT_BASE":      "Reliable preferences and constraints",
    "MONOTONE_CONSTRAINED": "Reliable preferences with a relative constraint",
    "NOISY":                "20% of preference labels flipped at random",
    "LATE_SWITCHER":        "Hidden objective changes mid-run",
}

YLABEL = r"Simple regret $SR_t$"
XLABEL = "Iteration"


def plot_persona_panel(
    ax,
    arm_curves: Dict[str, np.ndarray],
    persona: str,
    switch_iter: Optional[int] = None,
    band: str = "none",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    legend: bool = True,
) -> None:
    """arm_curves: {arm: array [n_seeds, n_iters]}.  band: "none" | "iqr"."""
    y_min, y_max, n_it = 0.0, 0.0, 1
    for k, arm in enumerate(ARM_ORDER):
        if arm not in arm_curves:
            continue
        arr = np.asarray(arm_curves[arm], dtype=float)
        n_it = max(n_it, arr.shape[1])
        it = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        st = ARM_STYLES[arm]
        ax.plot(it, mean, label=ARM_LABELS[arm],
                markevery=(k, 3), zorder=3 + k, **st)
        y_min, y_max = min(y_min, mean.min()), max(y_max, mean.max())
        if band == "iqr":
            lo, hi = np.percentile(arr, [25, 75], axis=0)
            ax.fill_between(it, lo, hi, color=st["color"], alpha=0.13,
                            lw=0, zorder=2)
            y_min, y_max = min(y_min, lo.min()), max(y_max, hi.max())

    # zero-regret reference (the optimum)
    ax.axhline(0.0, color="#999999", ls="-", lw=0.8, zorder=1)

    pad = 0.05 * (y_max - y_min or 1.0)
    ax.set_ylim(y_min - pad, y_max + 3.2 * pad)   # headroom for the legend
    ax.set_xlim(1, n_it)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    if switch_iter is not None:
        ax.axvline(switch_iter, color="#444444", ls=(0, (4, 3)), lw=1.1, zorder=2)
        ax.text(switch_iter + 0.4, ax.get_ylim()[1] * 0.965, "objective change",
                fontsize=8.5, va="top", ha="left", color="#444444")

    ax.set_title(PERSONA_TITLES.get(persona, persona), fontweight="bold", pad=20)
    desc = PERSONA_DESCRIPTIONS.get(persona)
    if desc:
        ax.text(0.5, 1.012, desc, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#666666")
    if show_xlabel:
        ax.set_xlabel(XLABEL)
    if show_ylabel:
        ax.set_ylabel(YLABEL)
    if legend:
        ax.legend(loc="upper right", frameon=False, borderaxespad=0.4)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _fig_subtitle(meta: dict, band: str) -> str:
    parts = [f"Mean simple regret across {meta.get('n_seeds', '?')} seeds",
             f"{meta.get('acqf', 'qEUBO')} acquisition"]
    if band == "iqr":
        parts.append("shaded band: interquartile range")
    return "  ·  ".join(parts)


def plot_all(
    results: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    meta: Optional[dict] = None,
    bands: Tuple[str, ...] = ("none", "iqr"),
) -> List[Path]:
    """One standalone figure per persona (no combined grid — the four images
    stay separate for slide-level flexibility), exported once per entry in
    `bands` ("iqr" files get a `_band` suffix).  Returns saved paths."""
    meta = meta or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for band in bands:
        suffix = "" if band == "none" else "_band"
        for persona, arm_curves in results.items():
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            sw = meta.get("switch_iter") if persona == "LATE_SWITCHER" else None
            plot_persona_panel(ax, arm_curves, persona, switch_iter=sw, band=band)
            fig.text(0.5, 0.008, _fig_subtitle(meta, band),
                     ha="center", fontsize=9, color="#666666")
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            p = out_dir / f"validator_{persona}{suffix}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# standalone re-plot from saved .npy files
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(Path(results_dir).glob("regret_*.npy")):
        stem = f.stem[len("regret_"):]                 # {PERSONA}_{arm}
        persona, arm = stem.rsplit("_", 1)
        results.setdefault(persona, {})[arm] = np.load(f)
    # stable, presentation-friendly persona order
    order = ["CONSISTENT_BASE", "MONOTONE_CONSTRAINED", "NOISY", "LATE_SWITCHER"]
    return {p: results[p] for p in order if p in results} | {
        p: v for p, v in results.items() if p not in order}


if __name__ == "__main__":
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent.parent / "experiments" / "experiments"
        / "results" / "final_comparison")
    res = load_results(rdir)
    if not res:
        sys.exit(f"No regret_*.npy files found in {rdir}")
    meta_path = rdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for p in plot_all(res, rdir, meta):
        print(f"Figure saved → {p}")
