"""
plot_validator_comparison.py
============================
Figures for the validator study: one panel per PERSONA, three curves per
panel (PBO / PBO+L2G / PBO+L2G+Validator), mean over seeds with an
inter-quartile band.  For the genuine late switcher a vertical dashed line
marks the utility switch (regret is dynamic — the spike at the switch is
expected; the story is the recovery slope after it).

Usable two ways:
    • called by run_validator_comparison.main() with results in memory;
    • standalone:  python plot_validator_comparison.py <results_dir>
      (re-plots from the saved regret_{persona}_{arm}.npy files).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")

ARM_LABELS = {
    "pbo":       "Standard PBO (preference only)",
    "l2g":       "PBO + L2G",
    "validated": "PBO + L2G + Validator",
}
ARM_STYLES = {
    "pbo":       dict(color="#666666", ls=":",  lw=2.0, marker="o", ms=4),
    "l2g":       dict(color="#c23b22", ls="--", lw=2.0, marker="s", ms=4),
    "validated": dict(color="#1a6faf", ls="-",  lw=2.6, marker="D", ms=4),
}
PERSONA_TITLES = {
    "CONSISTENT_BASE": "Consistent human",
    "NOISY":           "Noisy human (20% flips)",
    "LATE_SWITCHER":   "Late switcher (genuine change of mind)",
}


def plot_persona_panel(
    ax,
    arm_curves: Dict[str, np.ndarray],
    title: str,
    switch_iter: Optional[int] = None,
    band: str = "iqr",
) -> None:
    """arm_curves: {arm: array [n_seeds, n_iters]}."""
    for arm in ("pbo", "l2g", "validated"):
        if arm not in arm_curves:
            continue
        arr = np.asarray(arm_curves[arm], dtype=float)
        it = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        if band == "iqr":
            lo, hi = np.percentile(arr, [25, 75], axis=0)
        else:
            lo, hi = arr.min(axis=0), arr.max(axis=0)
        st = ARM_STYLES[arm]
        ax.plot(it, mean, label=f"{ARM_LABELS[arm]} (n={arr.shape[0]})", **st)
        ax.fill_between(it, lo, hi, color=st["color"], alpha=0.15, lw=0)

    if switch_iter is not None:
        ax.axvline(switch_iter, color="black", ls=":", lw=1.2)
        ax.text(switch_iter, ax.get_ylim()[1] * 0.97, " utility switch",
                fontsize=9, va="top", ha="left", color="black")

    ax.axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Simple regret  $SR_t = U^*_{\mathrm{active}}(t) - U_{\mathrm{active}(t)}(x_t^{best})$")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)


def plot_all(
    results: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    meta: Optional[dict] = None,
) -> List[Path]:
    """One figure per persona + one combined 1x3 grid.  Returns saved paths."""
    meta = meta or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acqf = meta.get("acqf", "qEUBO")
    saved: List[Path] = []

    for persona, arm_curves in results.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sw = meta.get("switch_iter") if persona == "LATE_SWITCHER" else None
        plot_persona_panel(ax, arm_curves,
                           PERSONA_TITLES.get(persona, persona), switch_iter=sw)
        fig.suptitle(f"Validator study — {acqf}", fontsize=11, y=0.995)
        fig.tight_layout()
        p = out_dir / f"validator_{persona}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    personas = list(results.keys())
    fig, axes = plt.subplots(1, len(personas), figsize=(6.2 * len(personas), 5.2))
    if len(personas) == 1:
        axes = [axes]
    for ax, persona in zip(axes, personas):
        sw = meta.get("switch_iter") if persona == "LATE_SWITCHER" else None
        plot_persona_panel(ax, results[persona],
                           PERSONA_TITLES.get(persona, persona), switch_iter=sw)
    sub = (f"acqf={acqf}   retest_delay={meta.get('retest_delay', '?')}   "
           f"seeds={meta.get('n_seeds', '?')}")
    fig.suptitle(f"PBO vs PBO+L2G vs PBO+L2G+Validator — {sub}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "validator_grid.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
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
    return results


if __name__ == "__main__":
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "experiments" / "results" / "validator_compare")
    res = load_results(rdir)
    if not res:
        sys.exit(f"No regret_*.npy files found in {rdir}")
    meta_path = rdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"switch_iter": 8}
    for p in plot_all(res, rdir, meta):
        print(f"Figure saved → {p}")
