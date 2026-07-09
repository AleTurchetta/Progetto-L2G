"""
validator.py
============
Preference VALIDATOR for preference-based BO under unreliable human feedback.

Paradigm shift vs robust_pref.py: NO preference is ever deleted.  Every
expressed comparison stays in the dataset forever.  Instead, the validator

  (1) AUDITS the preference history each iteration: after the PairwiseGP is
      fit, each comparison gets an agreement probability
          p = Phi( (mu_win - mu_lose) / sqrt(var_win + var_lose - 2 cov) )
      from the joint posterior.  A comparison the rest of the data contradicts
      gets p << 0.5 even though it is in the training set.

  (2) RE-TESTS the most incoherent comparison (p < tau_flag) by offering the
      SAME duel to the oracle again (order-swapped, no new simulation).
      Eligibility is gated by `retest_delay`: the duel happens at least
      `retest_delay` iterations after the preference was expressed
      (retest_delay=1 == "instant": earliest possible, since the flag itself
      needs one refit).

  (3) RESOLVES:
      * retest CONTRADICTS the original  -> likely a mistake (noisy):
          original kept at weight 1 (status OVERRULED), a corrective
          comparison is added at weight `correction_weight` (default 2).
          Net vote 2-1 for the truth; the surviving contradiction locally
          inflates the Laplace posterior variance ("higher uncertainty near
          that point").
      * retest CONFIRMS the original     -> genuine preference:
          weight bumped to `confirm_weight`; if `changepoint_confirmations`
          confirmed-incoherent events land within `changepoint_window`
          iterations, a preference CHANGE-POINT is declared at the first
          event: pre-change comparisons are capped at weight 1, post-change
          comparisons count `post_change_weight` times.  Nothing is deleted -
          the old data is outvoted where it conflicts, kept where it doesn't.

Weighting is implemented by replication (BoTorch's PairwiseGP exposes no
per-comparison noise), so `dataset()` returns a plain list of (x_win, x_lose)
pairs that drops into the existing PairwiseGP fitting code unchanged.

torch is imported lazily (only inside the posterior audit), so the pure
logic is unit-testable with numpy alone:  python validator.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# record
# ─────────────────────────────────────────────────────────────────────────────

UNVERIFIED = "unverified"   # never audited-suspicious
PENDING    = "pending"      # flagged incoherent, awaiting its validation duel
CONFIRMED  = "confirmed"    # re-tested, oracle repeated the original label
OVERRULED  = "overruled"    # re-tested, oracle contradicted the original label
RETEST     = "retest"       # a corrective comparison produced by a retest


@dataclass
class ComparisonRecord:
    rec_id: int
    x_win: Any                       # tensor/array [1, d] — the stated winner
    x_lose: Any
    m_win: Dict[str, float]         # stored metrics (already simulated)
    m_lose: Dict[str, float]
    iter_added: int
    weight: int = 1
    status: str = UNVERIFIED
    n_retests: int = 0
    retest_of: Optional[int] = None  # rec_id of the record this corrects
    last_p: Optional[float] = None   # last audit agreement probability


# ─────────────────────────────────────────────────────────────────────────────
# pure-math helper (numpy-testable)
# ─────────────────────────────────────────────────────────────────────────────

def agreement_from_stats(mu_win: float, mu_lose: float,
                         var_win: float, var_lose: float,
                         cov: float) -> float:
    """P[f(x_win) > f(x_lose)] under a joint Gaussian posterior."""
    denom = math.sqrt(max(var_win + var_lose - 2.0 * cov, 1e-12))
    z = (mu_win - mu_lose) / denom
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# validator
# ─────────────────────────────────────────────────────────────────────────────

class PreferenceValidator:
    """
    Owns the preference dataset for the validated PBO arm.

    Parameters
    ----------
    tau_flag : float
        Flag a comparison when the model's agreement probability drops below
        this value (0.25 = "the rest of the data says 3:1 this is wrong").
    min_iter : int
        No validation duels before this iteration (the GP needs data first).
    retest_delay : int
        A flagged comparison becomes duel-eligible `retest_delay` iterations
        after it was EXPRESSED.  1 = instant (earliest possible).  Raise to
        2-3 to model human test-retest spacing.
    max_retests_total / max_retests_per_pair : int
        Global and per-pair validation budgets.
    confirm_weight / correction_weight : int
        Replication weight of a confirmed original / of the corrective
        comparison added on overrule.
    changepoint_confirmations / changepoint_window / post_change_weight :
        Declare a preference change-point when `changepoint_confirmations`
        confirmed-incoherent events fall within `changepoint_window`
        iterations of each other; afterwards post-change comparisons enter
        the dataset `post_change_weight` times, pre-change ones are capped
        at 1.
    """

    def __init__(
        self,
        tau_flag: float = 0.25,
        min_iter: int = 5,
        retest_delay: int = 1,
        max_retests_total: int = 5,
        max_retests_per_pair: int = 1,
        confirm_weight: int = 2,
        correction_weight: int = 2,
        changepoint_confirmations: int = 2,
        changepoint_window: int = 5,
        post_change_weight: int = 2,
        verbose: bool = False,
    ):
        self.tau_flag = tau_flag
        self.min_iter = min_iter
        self.retest_delay = retest_delay
        self.max_retests_total = max_retests_total
        self.max_retests_per_pair = max_retests_per_pair
        self.confirm_weight = confirm_weight
        self.correction_weight = correction_weight
        self.changepoint_confirmations = changepoint_confirmations
        self.changepoint_window = changepoint_window
        self.post_change_weight = post_change_weight
        self.verbose = verbose

        self.records: List[ComparisonRecord] = []
        self.changepoint_iter: Optional[int] = None
        self.confirmed_events: List[int] = []       # iter_added of confirmed recs
        self.n_retests_done: int = 0
        self.incumbent_rec_id: Optional[int] = None  # record that promoted incumbent
        self._next_id: int = 0
        self.event_log: List[dict] = []              # for plots / debugging

    # -- data entry -----------------------------------------------------------

    def add_comparison(self, x_win, x_lose, m_win, m_lose, iteration: int,
                       weight: int = 1, retest_of: Optional[int] = None
                       ) -> ComparisonRecord:
        rec = ComparisonRecord(
            rec_id=self._next_id, x_win=x_win, x_lose=x_lose,
            m_win=m_win, m_lose=m_lose, iter_added=iteration,
            weight=weight,
            status=RETEST if retest_of is not None else UNVERIFIED,
            retest_of=retest_of,
        )
        self._next_id += 1
        self.records.append(rec)
        return rec

    # -- export for PairwiseGP fitting ----------------------------------------

    def dataset(self) -> List[Tuple[Any, Any]]:
        """(x_win, x_lose) pairs, replicated by effective weight. Never empty
        because of deletion — only reweighted."""
        out: List[Tuple[Any, Any]] = []
        for r in self.records:
            w = r.weight
            if self.changepoint_iter is not None:
                if r.iter_added >= self.changepoint_iter:
                    w = max(w, self.post_change_weight)
                else:
                    w = min(w, 1)
            out.extend([(r.x_win, r.x_lose)] * int(w))
        return out

    # -- audit ----------------------------------------------------------------

    def _agreement(self, pref_model, rec: ComparisonRecord) -> float:
        import torch
        X = torch.cat([rec.x_win, rec.x_lose], dim=0)
        with torch.no_grad():
            post = pref_model.posterior(X)
            mean = post.mean.reshape(-1)
            cov = post.mvn.covariance_matrix.reshape(2, 2)
        return agreement_from_stats(
            float(mean[0]), float(mean[1]),
            float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1]),
        )

    def audit(self, pref_model, iteration: int) -> Optional[ComparisonRecord]:
        """
        Score every auditable record, update pending flags, and return the
        record whose validation duel should happen THIS iteration (or None).
        """
        if pref_model is None:
            return None

        flagged: List[ComparisonRecord] = []
        for r in self.records:
            if r.status not in (UNVERIFIED, PENDING):
                continue
            if r.retest_of is not None:                 # never audit corrections
                continue
            if r.n_retests >= self.max_retests_per_pair:
                continue
            p = self._agreement(pref_model, r)
            r.last_p = p
            if p < self.tau_flag:
                if r.status != PENDING and self.verbose:
                    print(f"  [validator] flag rec#{r.rec_id} (it={r.iter_added}) "
                          f"p={p:.3f} < tau={self.tau_flag}")
                r.status = PENDING
                flagged.append(r)
            elif r.status == PENDING:
                r.status = UNVERIFIED                   # model no longer objects
                if self.verbose:
                    print(f"  [validator] unflag rec#{r.rec_id} (p={p:.3f})")

        if iteration < self.min_iter:
            return None
        if self.n_retests_done >= self.max_retests_total:
            return None

        eligible = [r for r in flagged
                    if (iteration - r.iter_added) >= self.retest_delay]
        if not eligible:
            return None
        return min(eligible, key=lambda r: r.last_p)

    # -- resolution ------------------------------------------------------------

    def resolve(self, rec: ComparisonRecord, confirmed: bool, iteration: int
                ) -> Optional[ComparisonRecord]:
        """
        Apply the outcome of a validation duel.  Returns the corrective record
        on overrule, else None.  Never deletes anything.
        """
        rec.n_retests += 1
        self.n_retests_done += 1

        if confirmed:
            rec.status = CONFIRMED
            rec.weight = max(rec.weight, self.confirm_weight)
            self.confirmed_events.append(rec.iter_added)
            self._maybe_declare_changepoint()
            self.event_log.append({"iteration": iteration, "event": "confirmed",
                                   "rec_id": rec.rec_id,
                                   "changepoint": self.changepoint_iter})
            if self.verbose:
                print(f"  [validator] rec#{rec.rec_id} CONFIRMED "
                      f"(weight->{rec.weight}, changepoint={self.changepoint_iter})")
            return None

        rec.status = OVERRULED                          # kept at its weight
        corr = self.add_comparison(
            rec.x_lose, rec.x_win, rec.m_lose, rec.m_win,
            iteration, weight=self.correction_weight, retest_of=rec.rec_id,
        )
        self.event_log.append({"iteration": iteration, "event": "overruled",
                               "rec_id": rec.rec_id, "correction_id": corr.rec_id})
        if self.verbose:
            print(f"  [validator] rec#{rec.rec_id} OVERRULED "
                  f"(kept w={rec.weight}; correction rec#{corr.rec_id} w={corr.weight})")
        return corr

    def _maybe_declare_changepoint(self) -> None:
        if self.changepoint_iter is not None:
            return
        k = self.changepoint_confirmations
        if len(self.confirmed_events) < k:
            return
        ev = sorted(self.confirmed_events)
        for i in range(len(ev) - k + 1):
            if ev[i + k - 1] - ev[i] <= self.changepoint_window:
                self.changepoint_iter = ev[i]
                if self.verbose:
                    print(f"  [validator] CHANGE-POINT declared at iteration "
                          f"{self.changepoint_iter}")
                return

    # -- incumbent guard --------------------------------------------------------

    def note_incumbent_promotion(self, rec: ComparisonRecord) -> None:
        """Call when a fresh preference moves the incumbent to its winner."""
        self.incumbent_rec_id = rec.rec_id

    def incumbent_suspect(self) -> bool:
        """True while the preference holding the incumbent is flagged-pending
        or has been overruled (loop should fall back to posterior-argmax)."""
        if self.incumbent_rec_id is None:
            return False
        for r in self.records:
            if r.rec_id == self.incumbent_rec_id:
                return r.status in (PENDING, OVERRULED)
        return False

    def clear_incumbent_link(self) -> None:
        self.incumbent_rec_id = None

    # -- introspection -----------------------------------------------------------

    def summary(self) -> dict:
        by = {}
        for r in self.records:
            by[r.status] = by.get(r.status, 0) + 1
        return {"n_records": len(self.records), "by_status": by,
                "retests_done": self.n_retests_done,
                "changepoint_iter": self.changepoint_iter}


# ─────────────────────────────────────────────────────────────────────────────
# self-test (numpy only — no torch/botorch needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=== validator self-test (logic only) ===")

    good = np.array([[0.2, 0.2]])
    bad = np.array([[0.9, 0.9]])
    mg = {"tracking_mse": 0.1}
    mb = {"tracking_mse": 0.9}

    # agreement math: strong positive gap -> p ~ 1; negative gap -> p ~ 0
    assert agreement_from_stats(1.0, -1.0, 0.1, 0.1, 0.0) > 0.99
    assert agreement_from_stats(-1.0, 1.0, 0.1, 0.1, 0.0) < 0.01
    assert abs(agreement_from_stats(0.0, 0.0, 0.1, 0.1, 0.0) - 0.5) < 1e-9
    print("agreement_from_stats ✓")

    # overrule keeps original + adds weighted correction; nothing deleted
    v = PreferenceValidator(retest_delay=1)
    r = v.add_comparison(bad, good, mb, mg, iteration=8)     # corrupted: bad "wins"
    corr = v.resolve(r, confirmed=False, iteration=9)
    assert r.status == OVERRULED and r.weight == 1
    assert corr.retest_of == r.rec_id and corr.weight == 2
    ds = v.dataset()
    assert len(ds) == 3                                       # 1 wrong + 2 right
    n_right = sum(1 for (w, _l) in ds if w is good)
    assert n_right == 2
    print("overrule: kept 1, corrected x2, net majority right ✓")

    # confirm bumps weight; two confirms in window declare change-point
    v2 = PreferenceValidator(changepoint_confirmations=2, changepoint_window=5)
    a = v2.add_comparison(good, bad, mg, mb, iteration=3)     # old-utility comp
    b = v2.add_comparison(bad, good, mb, mg, iteration=8)     # post-switch comp
    c = v2.add_comparison(bad, good, mb, mg, iteration=10)
    v2.resolve(b, confirmed=True, iteration=9)
    assert v2.changepoint_iter is None and b.weight == 2
    v2.resolve(c, confirmed=True, iteration=11)
    assert v2.changepoint_iter == 8, v2.changepoint_iter
    ds2 = v2.dataset()
    # pre-change 'a' capped at 1; post-change b,c at weight 2 each -> 1+2+2
    assert len(ds2) == 5, len(ds2)
    print("confirm + change-point reweighting ✓")

    # confirmations far apart do NOT declare a change-point
    v3 = PreferenceValidator(changepoint_confirmations=2, changepoint_window=5)
    e1 = v3.add_comparison(bad, good, mb, mg, iteration=2)
    e2 = v3.add_comparison(bad, good, mb, mg, iteration=12)
    v3.resolve(e1, True, 3)
    v3.resolve(e2, True, 13)
    assert v3.changepoint_iter is None
    print("isolated confirmations: no change-point ✓")

    # incumbent guard
    v4 = PreferenceValidator()
    r4 = v4.add_comparison(bad, good, mb, mg, iteration=6)
    v4.note_incumbent_promotion(r4)
    assert not v4.incumbent_suspect()
    r4.status = PENDING
    assert v4.incumbent_suspect()
    v4.resolve(r4, confirmed=False, iteration=7)
    assert v4.incumbent_suspect()          # overruled -> still suspect until reset
    v4.clear_incumbent_link()
    assert not v4.incumbent_suspect()
    print("incumbent guard ✓")

    # budgets: per-pair and total
    v5 = PreferenceValidator(max_retests_total=1)
    p1 = v5.add_comparison(bad, good, mb, mg, iteration=6)
    v5.resolve(p1, confirmed=True, iteration=7)
    assert v5.n_retests_done == 1 and p1.n_retests == 1
    print("budgets tracked ✓")

    print("=== all validator logic tests passed ===")
