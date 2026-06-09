"""
robust_pref.py
==============
Robustness layer for preference-based BO under adversarial feedback
(NOISY random flips, LATE_SWITCHER preference inversion).

Diagnosis it addresses
----------------------
Our acqf comparison showed the regret is driven by the *incumbent rule*
(`best_idx = whoever was preferred last`) and by feeding every (possibly
corrupted) comparison into the PairwiseGP with equal trust.  This module
provides three composable fixes:

  (1) robust_incumbent      — pick the incumbent as the argmax of the
                              preference-GP posterior mean over ALL evaluated
                              points, so one bad flip can't move it.
  (2) clean_comparisons     — drop/repair comparisons that are likely corrupted:
                                • trust-but-verify: drop a comparison whose
                                  "winner" the (uncorrupted) metric surrogates
                                  predict to be worse than its "loser";
                                • contradiction de-dup: if the same pair appears
                                  with opposite directions, keep the majority.
  (3) build_robust_pref_model — clean, then fit a PairwiseGP with a softened
                                (noise-tolerant) jitter.

Design notes
------------
• Pure-logic parts (clean_comparisons) work on numpy OR torch and are unit
  tested below without torch/botorch.
• torch / botorch are imported lazily, so this module imports anywhere.
• Grounding modes:
    "pareto"  — realistic & target-agnostic: drop a comparison only if the loser
                Pareto-dominates the winner on the predicted metrics
                (all metrics lower-or-equal, at least one strictly lower).
    "utility" — oracle-informed (peeks at the target utility): drop if the
                winner's predicted utility is below the loser's.  Stronger
                (rescues LATE_SWITCHER) but optimistic — use to bound what
                trust-but-verify can achieve, then validate with "pareto".
    "off"     — no grounding (de-dup only).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

METRIC_KEYS_DEFAULT = ("overshoot_pct", "settling_time", "tracking_mse")


# ─────────────────────────────────────────────────────────────────────────────
# small array helpers (tensor-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _scalar(v) -> float:
    return float(_to_np(v).reshape(-1)[0])


def _point_key(x, ndigits: int = 5) -> Tuple[float, ...]:
    return tuple(np.round(_to_np(x).reshape(-1), ndigits).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# (2) comparison cleaning — trust-but-verify + contradiction de-dup
# ─────────────────────────────────────────────────────────────────────────────

def _predict_metrics(
    x,
    surrogate_wrappers: Dict[str, Callable],
    metric_keys: Sequence[str],
) -> Dict[str, float]:
    """Predict real-unit metrics for a single point using the L2G surrogate wrappers."""
    return {k: _scalar(surrogate_wrappers[k](x)) for k in metric_keys}


def _utility_of(metrics: Dict[str, float], utility) -> float:
    if utility is None:
        raise ValueError("grounding='utility' requires a utility object/callable.")
    return utility.utility(metrics) if hasattr(utility, "utility") else utility(metrics)


def _is_corrupted(
    m_win: Dict[str, float],
    m_lose: Dict[str, float],
    *,
    grounding: str,
    utility,
    margin: float,
    metric_keys: Sequence[str],
) -> bool:
    """True if the metric surrogates contradict the stated preference."""
    if grounding == "utility":
        return _utility_of(m_win, utility) < _utility_of(m_lose, utility) - margin
    if grounding == "pareto":
        # loser dominates winner (all metrics <=, at least one <)  → suspicious
        le = all(m_lose[k] <= m_win[k] + margin for k in metric_keys)
        lt = any(m_lose[k] < m_win[k] - margin for k in metric_keys)
        return le and lt
    return False


def _resolve_contradictions(
    pairs: List[Tuple],
) -> List[Tuple]:
    """
    Collapse contradictory repeats of the SAME unordered pair.

    For each unordered pair of points, tally how often each point won; keep the
    majority direction with its net count, drop a tie entirely.  Pairs that never
    repeat (the common case) pass through unchanged.
    """
    groups: Dict[frozenset, Dict] = {}
    order: List[frozenset] = []
    for (xw, xl) in pairs:
        kw, kl = _point_key(xw), _point_key(xl)
        key = frozenset((kw, kl))
        if key not in groups:
            groups[key] = {"votes": {}, "rep": {}}
            order.append(key)
        groups[key]["votes"][kw] = groups[key]["votes"].get(kw, 0) + 1
        groups[key]["rep"][kw] = xw
        groups[key]["rep"][kl] = xl

    out: List[Tuple] = []
    for key in order:
        votes = groups[key]["votes"]
        rep = groups[key]["rep"]
        pts = list(frozenset(rep.keys()))
        if len(pts) == 1:                       # degenerate (xw==xl); skip
            continue
        a, b = pts[0], pts[1]
        va, vb = votes.get(a, 0), votes.get(b, 0)
        if va == vb:                            # contradictory tie → drop pair
            continue
        winner_key, loser_key = (a, b) if va > vb else (b, a)
        net = abs(va - vb)
        for _ in range(net):                    # preserve majority strength
            out.append((rep[winner_key], rep[loser_key]))
    return out


def clean_comparisons(
    pref_data: List[Tuple],
    *,
    surrogate_wrappers: Optional[Dict[str, Callable]] = None,
    utility=None,
    grounding: str = "pareto",
    ground_margin: float = 0.0,
    dedup: bool = True,
    metric_keys: Sequence[str] = METRIC_KEYS_DEFAULT,
    verbose: bool = False,
) -> List[Tuple]:
    """
    Return a cleaned list of (x_win, x_lose) comparisons.

    Parameters
    ----------
    pref_data : list of (x_win, x_lose)
        Raw comparisons from L2GEngine.get_preference_dataset().
    surrogate_wrappers : {metric: callable(X)->real value}
        The L2G metric surrogates (l2g.sub_surrogates).  Required for grounding.
    utility : object with .utility(dict)->float, or callable
        Needed only for grounding="utility".
    grounding : "pareto" | "utility" | "off"
    dedup : bool
        Resolve contradictory repeats of the same pair.
    """
    kept: List[Tuple] = []
    n_dropped = 0
    do_ground = grounding != "off" and surrogate_wrappers is not None
    for (xw, xl) in pref_data:
        if do_ground:
            m_win = _predict_metrics(xw, surrogate_wrappers, metric_keys)
            m_lose = _predict_metrics(xl, surrogate_wrappers, metric_keys)
            if _is_corrupted(m_win, m_lose, grounding=grounding, utility=utility,
                             margin=ground_margin, metric_keys=metric_keys):
                n_dropped += 1
                continue
        kept.append((xw, xl))

    if dedup:
        before = len(kept)
        kept = _resolve_contradictions(kept)
        if verbose and before != len(kept):
            print(f"  [robust_pref] de-dup removed {before - len(kept)} contradictory comp(s)")

    if verbose and n_dropped:
        print(f"  [robust_pref] grounding ({grounding}) dropped {n_dropped} "
              f"comparison(s) that contradict the metric surrogates")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# (3) noise-tolerant PairwiseGP build  (lazy torch/botorch)
# ─────────────────────────────────────────────────────────────────────────────

def build_robust_pref_model(
    pref_data: List[Tuple],
    *,
    surrogate_wrappers: Optional[Dict[str, Callable]] = None,
    utility=None,
    grounding: str = "pareto",
    ground_margin: float = 0.0,
    dedup: bool = True,
    jitter: float = 1e-2,
    metric_keys: Sequence[str] = METRIC_KEYS_DEFAULT,
    verbose: bool = False,
):
    """
    Clean the comparisons, then fit a PairwiseGP with a softened jitter
    (noise tolerance).  Returns the fitted model, or None if no usable
    comparisons remain.

    Drop-in replacement for the inline PairwiseGP block in run_loop.
    """
    clean = clean_comparisons(
        pref_data, surrogate_wrappers=surrogate_wrappers, utility=utility,
        grounding=grounding, ground_margin=ground_margin, dedup=dedup,
        metric_keys=metric_keys, verbose=verbose,
    )
    if not clean:
        return None

    import torch
    from botorch.models.pairwise_gp import (
        PairwiseGP, PairwiseLaplaceMarginalLogLikelihood,
    )
    from botorch.fit import fit_gpytorch_mll

    X_win = torch.cat([p[0] for p in clean], dim=0)
    X_lose = torch.cat([p[1] for p in clean], dim=0)
    train_X = torch.cat([X_win, X_lose], dim=0)
    M = X_win.shape[0]
    comps = torch.tensor([[i, M + i] for i in range(M)], dtype=torch.long)

    # Softened jitter ≈ observation noise on the latent → tolerates inconsistent
    # comparisons (random flips).  Fall back if the kwarg name differs by version.
    try:
        model = PairwiseGP(datapoints=train_X, comparisons=comps, jitter=jitter)
    except TypeError:
        model = PairwiseGP(datapoints=train_X, comparisons=comps)

    fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# (1) robust incumbent selection  (lazy torch)
# ─────────────────────────────────────────────────────────────────────────────

def robust_incumbent(pref_model, train_x, fallback_idx: Optional[int] = None) -> int:
    """
    Incumbent = argmax of the preference-GP posterior mean over ALL evaluated
    points.  Robust to a single flipped comparison, because it integrates the
    whole (already-cleaned) preference posterior instead of trusting the last
    label.

    Returns `fallback_idx` (e.g. the current best_idx) when no preference model
    exists yet (cold start).
    """
    if pref_model is None:
        if fallback_idx is None:
            raise ValueError("robust_incumbent needs a fallback_idx when pref_model is None.")
        return int(fallback_idx)

    import torch
    with torch.no_grad():
        mean = pref_model.posterior(train_x).mean.reshape(-1)
    return int(torch.argmax(mean).item())


# ─────────────────────────────────────────────────────────────────────────────
# self-test (numpy only — no torch/botorch needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== robust_pref self-test (logic only) ===")

    # points in 1-D "design" space; wrappers map x -> metric (here metric == |x|-ish)
    def W_overshoot(x): return float(_to_np(x).reshape(-1)[0]) * 10.0
    def W_settling(x):  return float(_to_np(x).reshape(-1)[0]) * 5.0
    def W_mse(x):       return float(_to_np(x).reshape(-1)[0]) * 1.0
    wrappers = {"overshoot_pct": W_overshoot, "settling_time": W_settling, "tracking_mse": W_mse}

    good = np.array([[0.2, 0.2]])   # low metrics  → better
    bad  = np.array([[0.9, 0.9]])   # high metrics → worse

    # honest comparison: good beats bad → should be kept
    honest = [(good, bad)]
    # corrupted comparison: bad beats good → grounding should DROP it
    corrupt = [(bad, good)]

    # pareto grounding: 'good' lower on all metrics, so loser(good) dominates winner(bad) → drop
    assert len(clean_comparisons(honest, surrogate_wrappers=wrappers, grounding="pareto")) == 1
    assert len(clean_comparisons(corrupt, surrogate_wrappers=wrappers, grounding="pareto")) == 0
    print("pareto grounding: keeps honest, drops corrupted ✓")

    # utility grounding with a simple target-seeking-ish utility (lower metrics better)
    class U:
        def utility(self, m): return -(m["overshoot_pct"] + m["settling_time"] + m["tracking_mse"])
    assert len(clean_comparisons(honest, surrogate_wrappers=wrappers, utility=U(), grounding="utility")) == 1
    assert len(clean_comparisons(corrupt, surrogate_wrappers=wrappers, utility=U(), grounding="utility")) == 0
    print("utility grounding: keeps honest, drops corrupted ✓")

    # contradiction de-dup: same pair, 2 say good>bad, 1 says bad>good → keep net 1 (good>bad)
    mixed = [(good, bad), (good, bad), (bad, good)]
    out = clean_comparisons(mixed, grounding="off", dedup=True)
    assert len(out) == 1 and _point_key(out[0][0]) == _point_key(good), out
    print("contradiction de-dup: majority kept, minority cancelled ✓")

    # tie → dropped
    tie = [(good, bad), (bad, good)]
    assert len(clean_comparisons(tie, grounding="off", dedup=True)) == 0
    print("contradiction de-dup: tie dropped ✓")

    # grounding='off' with no wrappers leaves honest data untouched
    assert len(clean_comparisons(honest, grounding="off", dedup=False)) == 1
    print("grounding off: passthrough ✓")

    print("=== all robust_pref logic tests passed ===")
