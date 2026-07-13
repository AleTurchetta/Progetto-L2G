"""
base_oracle.py


1. One standardized, target-seeking utility shared by every persona.
   U(x) = - || normalize(metrics) - normalize(target) ||_{w,2}
   The target is a *reachable* point: we sample a random (Kp, Ki), simulate it,
   and use its (overshoot, settling, MSE) as the target.  Because the target is
   achievable, U = 0 is attainable and Simple Regret is bounded in [0, 1].

2. A single consistent "base" oracle that gives detailed,
   non-contradictory feedback: a preference (by utility) PLUS directional
   constraints that nudge the next candidate toward the target without ever
   excluding the current winner (so the feasible set is never empty).

3. Seven adversarial personas that merely CORRUPT the base output.
   They all share the same utility instance, so their regret curves are
   directly comparable to the base oracle's — the only thing that changes is
   the feedback channel, never the hidden objective.

The output of every .query() is a plain dict matching the L2GOutput
schema, so it can be passed straight to `L2GEngine.process_feedback(...,
simulated_output=...)` exactly like the old oracle.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 0.  METRIC NORMALISATION  (shared, fixed — identical for every persona)
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS: Tuple[str, str, str] = ("overshoot_pct", "settling_time", "tracking_mse")

# Per-metric scales map raw units → roughly [0, 1].  Deviations are clipped so a
# single unstable candidate (overshoot >> 100, settling = 50, MSE = 1e5) can no
# longer dominate the regret axis the way it did in the original plots.
T_MAX: float = 40.0
_SCALES: Dict[str, float] = {
    "overshoot_pct": 100.0,   # % overshoot
    "settling_time": T_MAX,   # seconds (unstable cap = T_MAX + 10 → clips to 1)
    "tracking_mse":  2.0,     # stable MSE is typically < 1; unstable clips to 1
}


def normalize_metrics(metrics: Dict[str, float]) -> np.ndarray:
    """Map a metrics dict to a clipped, normalised 3-vector in [0, 1]^3."""
    out = np.empty(3, dtype=float)
    for i, k in enumerate(METRIC_KEYS):
        v = metrics.get(k, np.nan)
        if v is None or np.isnan(v):
            v = _SCALES[k]            # missing/NaN → worst (normalised 1.0)
        out[i] = np.clip(v / _SCALES[k], 0.0, 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TARGET-SEEKING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

class TargetSeekingUtility:
    """
    Normalised weighted-L2 utility around a (reachable) target metric profile.

        U(m) = - sqrt( Σ_i w_i ((n_i(m) - n_i(target))^2) / Σ_i w_i )

    U ∈ [-1, 0];  U = 0 exactly at the target.  Higher is better.

    Parameters
    ----------
    target : dict
        Target metrics {overshoot_pct, settling_time, tracking_mse}.
    weights : (w_ov, w_ts, w_mse)
        Per-metric importance.  Default equal weights.  **Kept identical across
        every persona** in a comparison so that U* and the landscape match.
    """

    def __init__(
        self,
        target: Dict[str, float],
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.target = dict(target)
        self.weights = np.asarray(weights, dtype=float)
        self._wsum = float(self.weights.sum()) + 1e-12
        self._target_norm = normalize_metrics(self.target)

    def utility(self, metrics: Dict[str, float]) -> float:
        diff = normalize_metrics(metrics) - self._target_norm
        d = np.sqrt(float((self.weights * diff ** 2).sum()) / self._wsum)
        return float(-d)

    def __repr__(self) -> str:
        t = self.target
        return (
            f"TargetSeekingUtility(target=[ov={t['overshoot_pct']:.2f}%, "
            f"ts={t['settling_time']:.2f}s, mse={t['tracking_mse']:.4f}], "
            f"w={tuple(self.weights)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PERSONA REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class Persona(Enum):
    CONSISTENT_BASE    = "CONSISTENT_BASE"     # the new gold-standard reference
    NOISY              = "NOISY"               # random preference flips
    CONTRADICTORY      = "CONTRADICTORY"       # adds a constraint the winner violates
    DRASTIC_ABSOLUTE   = "DRASTIC_ABSOLUTE"    # hard absolute thresholds at the target
    DRASTIC_RELATIVE   = "DRASTIC_RELATIVE"    # "better than A on every metric"
    DIRECTION_ONLY     = "DIRECTION_ONLY"      # strips the preference label
    LATE_SWITCHER      = "LATE_SWITCHER"       # inverts preference after k iters
    AMBIGUOUS_POSITIVE = "AMBIGUOUS_POSITIVE"  # always "prefer B"


# Convenience groupings used by the experiment driver / plots.
ADVERSARIAL_PERSONAS: Tuple[Persona, ...] = (
    Persona.NOISY,
    Persona.CONTRADICTORY,
    Persona.DRASTIC_ABSOLUTE,
    Persona.DRASTIC_RELATIVE,
    Persona.DIRECTION_ONLY,
    Persona.LATE_SWITCHER,
    Persona.AMBIGUOUS_POSITIVE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONSTRAINT HELPERS  (emit plain dicts matching the L2GOutput schema)
# ─────────────────────────────────────────────────────────────────────────────

def _directional_vs(ref_label: str, metric: str, iteration: int,
                    margin: float = 0.02, confidence: float = 0.9) -> dict:
    """'next candidate must be no worse than `ref_label` on `metric`'."""
    return {
        "id": f"dir_{metric}_vs_{ref_label}_it{iteration}",
        "subfunction_id": metric,
        "constraint_type": "directional_improvement",
        "operator": "<=",
        "reference_candidate": ref_label,
        "threshold": None,
        "margin": float(margin),
        "weight": 1.0,
        "confidence": float(confidence),
    }


def _absolute(metric: str, threshold: float, iteration: int,
              weight: float = 1.0, confidence: float = 0.9) -> dict:
    """'metric <= threshold' (hard absolute bound)."""
    return {
        "id": f"abs_{metric}_it{iteration}",
        "subfunction_id": metric,
        "constraint_type": "upper_bound_absolute",
        "operator": "<=",
        "reference_candidate": None,
        "threshold": float(threshold),
        "margin": 0.0,
        "weight": float(weight),
        "confidence": float(confidence),
    }


def _empty_output() -> dict:
    return {"feedback_type": "none", "preference": None, "constraints": []}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BASE ORACLE  (consistent, detailed, helpful)
# ─────────────────────────────────────────────────────────────────────────────

class BaseOracle:
    """
    The consistent standard persona.

    Feedback each iteration:
      • a clear preference for the higher-utility candidate, with confidence
        that grows with the utility gap, and
      • directional constraints telling the optimiser to not regress, relative
        to the winner, on every metric that is still above its target value.

    This feedback is always self-consistent and always feasible (the winner
    itself satisfies the constraints), so it gives a clean upper bound on how
    fast each acquisition function can move.
    """

    persona = Persona.CONSISTENT_BASE

    def __init__(
        self,
        utility: TargetSeekingUtility,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.utility_fn = utility
        self.seed = seed
        self.verbose = verbose
        self._rng = random.Random(seed)
        self.history: List[dict] = []

    # -- shared utility (delegates so every persona scores with the SAME U) ----
    def utility(self, metrics: Dict[str, float]) -> float:
        return self.utility_fn.utility(metrics)

    # -- the un-corrupted, "honest" output -------------------------------------
    def _honest_output(self, mA: dict, mB: dict, iteration: int) -> dict:
        uA, uB = self.utility(mA), self.utility(mB)
        if uB > uA:
            winner, loser, win_m = "B", "A", mB
        else:
            winner, loser, win_m = "A", "B", mA
        confidence = float(np.clip(0.6 + 5.0 * abs(uB - uA), 0.0, 1.0))

        # Detailed but feasible constraints: don't regress (vs winner) on any
        # metric that still sits above the target.
        target = self.utility_fn.target
        constraints: List[dict] = []
        for metric in METRIC_KEYS:
            if win_m.get(metric, np.inf) > target[metric] + 1e-9:
                constraints.append(_directional_vs(winner, metric, iteration))

        return {
            "feedback_type": "both" if constraints else "preference_only",
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": round(confidence, 3),
            },
            "constraints": constraints,
        }

    # -- public entry point ----------------------------------------------------
    def query(self, mA: dict, mB: dict, iteration: int = 1) -> dict:
        out = self._honest_output(mA, mB, iteration)
        self._finalise(out, mA, mB, iteration)
        return out

    # -- bookkeeping / logging -------------------------------------------------
    def _finalise(self, out: dict, mA: dict, mB: dict, iteration: int) -> None:
        self.history.append({"iteration": iteration, "output": out})
        if self.verbose:
            pref = out.get("preference")
            w = pref["preferred_candidate"] if pref else "—"
            print(
                f"  [{self.persona.value}] it={iteration} type={out['feedback_type']} "
                f"winner={w} U(A)={self.utility(mA):.4f} U(B)={self.utility(mB):.4f} "
                f"constraints={len(out['constraints'])}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ADVERSARIAL PERSONAS  (corrupt the base output; share the same utility)
# ─────────────────────────────────────────────────────────────────────────────

class MonotoneConstrainedOracle(BaseOracle):
    """S1b port: honest preference + one mild, always-satisfiable relative
    constraint (no tracking_mse regression vs A, 5% margin)."""
    persona = Persona.CONSISTENT_BASE  # logging label only

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        out["constraints"] = [_directional_vs("A", "tracking_mse", iteration,
                                              margin=0.05, confidence=0.9)]
        out["feedback_type"] = "both"
        self._finalise(out, mA, mB, iteration)
        return out

class NoisyOracle(BaseOracle):
    """S2 — flips the preference label with probability `noise_level`."""
    persona = Persona.NOISY

    def __init__(self, utility, seed=42, verbose=False, noise_level: float = 0.20):
        super().__init__(utility, seed, verbose)
        self.noise_level = noise_level

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        pref = out.get("preference")
        if pref and self._rng.random() < self.noise_level:
            pref["preferred_candidate"], pref["other_candidate"] = (
                pref["other_candidate"], pref["preferred_candidate"]
            )
            pref["confidence"] = round(max(0.1, pref["confidence"] - 0.2), 3)
            if self.verbose:
                print(f"  [{self.persona.value}] noise flip (p={self.noise_level:.0%})")
        self._finalise(out, mA, mB, iteration)
        return out


class LateSwitcherOracle(BaseOracle):
    """S7 — honest for the first `switch_iter`-1 iters, then inverts the label."""
    persona = Persona.LATE_SWITCHER

    def __init__(self, utility, seed=42, verbose=False, switch_iter: int = 4):
        super().__init__(utility, seed, verbose)
        self.switch_iter = switch_iter

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        pref = out.get("preference")
        if pref and iteration >= self.switch_iter:
            pref["preferred_candidate"], pref["other_candidate"] = (
                pref["other_candidate"], pref["preferred_candidate"]
            )
            if self.verbose:
                print(f"  [{self.persona.value}] preference inverted (it>={self.switch_iter})")
        self._finalise(out, mA, mB, iteration)
        return out


class AmbiguousPositiveOracle(BaseOracle):
    """S8 — always claims to prefer B, regardless of true utility; no constraints."""
    persona = Persona.AMBIGUOUS_POSITIVE

    def query(self, mA, mB, iteration=1):
        out = {
            "feedback_type": "preference_only",
            "preference": {"preferred_candidate": "B", "other_candidate": "A",
                           "confidence": 0.9},
            "constraints": [],
        }
        self._finalise(out, mA, mB, iteration)
        return out


class ContradictoryOracle(BaseOracle):
    """
    S3 — keeps the honest preference but adds a hard absolute constraint that the
    *preferred* candidate already violates, forcing the system toward infeasibility.
    """
    persona = Persona.CONTRADICTORY

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        pref = out.get("preference")
        if pref:
            win_m = mB if pref["preferred_candidate"] == "B" else mA
            # Pick the winner's worst (most normalised-above-target) metric.
            tnorm = self.utility_fn._target_norm
            wnorm = normalize_metrics(win_m)
            worst_i = int(np.argmax(wnorm - tnorm))
            worst_metric = METRIC_KEYS[worst_i]
            raw_val = win_m.get(worst_metric, _SCALES[worst_metric])
            tight = max(1e-3, raw_val * 0.7)   # below the winner's actual value
            out["constraints"] = [_absolute(worst_metric, tight, iteration,
                                            weight=1.0, confidence=0.9)]
            out["feedback_type"] = "both"
        self._finalise(out, mA, mB, iteration)
        return out


class DirectionOnlyOracle(BaseOracle):
    """S6 — never reveals a preference label; emits only the directional constraints."""
    persona = Persona.DIRECTION_ONLY

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        out["preference"] = None
        out["feedback_type"] = "direction_only" if out["constraints"] else "none"
        self._finalise(out, mA, mB, iteration)
        return out


class DrasticAbsoluteOracle(BaseOracle):
    """
    S4 — keeps the honest preference and adds tight hard absolute bounds taken
    directly from the target (overshoot & settling).  A narrow feasible region.
    """
    persona = Persona.DRASTIC_ABSOLUTE

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        t = self.utility_fn.target
        out["constraints"] = [
            _absolute("overshoot_pct", max(t["overshoot_pct"], 1e-3), iteration,
                      weight=1.5, confidence=0.95),
            _absolute("settling_time", max(t["settling_time"], 1e-3), iteration,
                      weight=1.0, confidence=0.9),
        ]
        out["feedback_type"] = "both"
        self._finalise(out, mA, mB, iteration)
        return out


class DrasticRelativeOracle(BaseOracle):
    """
    S5 — keeps the honest preference and demands the next candidate beat the
    incumbent A on *every* metric (directional improvement vs A, zero margin).
    """
    persona = Persona.DRASTIC_RELATIVE

    def query(self, mA, mB, iteration=1):
        out = self._honest_output(mA, mB, iteration)
        out["constraints"] = [
            _directional_vs("A", metric, iteration, margin=0.0, confidence=0.85)
            for metric in METRIC_KEYS
        ]
        out["feedback_type"] = "both"
        self._finalise(out, mA, mB, iteration)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FACTORY
# ─────────────────────────────────────────────────────────────────────────────

_PERSONA_CLASSES = {
    Persona.CONSISTENT_BASE:    BaseOracle,
    Persona.NOISY:              NoisyOracle,
    Persona.CONTRADICTORY:      ContradictoryOracle,
    Persona.DRASTIC_ABSOLUTE:   DrasticAbsoluteOracle,
    Persona.DRASTIC_RELATIVE:   DrasticRelativeOracle,
    Persona.DIRECTION_ONLY:     DirectionOnlyOracle,
    Persona.LATE_SWITCHER:      LateSwitcherOracle,
    Persona.AMBIGUOUS_POSITIVE: AmbiguousPositiveOracle,
}


def make_oracle(
    persona: Persona | str,
    utility: TargetSeekingUtility,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
) -> BaseOracle:
    """
    Build an oracle for `persona` bound to a shared `utility` instance.

    Extra keyword args are forwarded to the persona class, e.g.
        make_oracle(Persona.NOISY, util, noise_level=0.3)
        make_oracle(Persona.LATE_SWITCHER, util, switch_iter=5)
    """
    if isinstance(persona, str):
        persona = Persona(persona)
    cls = _PERSONA_CLASSES[persona]
    return cls(utility, seed=seed, verbose=verbose, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SIMULATION-BACKED HELPERS  (lazy import of control_utils)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_metrics(kp: float, ki: float, plant, t_final: float = T_MAX) -> dict:
    """Simulate one controller and return sanitised metrics (mirrors run_loop)."""
    import control_utils as cu
    t, y = cu.simulate_step(kp, ki, plant, t_final=t_final)
    m = cu.compute_metrics(t, y)
    if m["settling_time"] is None or np.isnan(m["settling_time"]):
        m["settling_time"] = t_final + 10.0
    if np.isnan(m["tracking_mse"]) or m["tracking_mse"] > 1e4:
        m["tracking_mse"] = 1e5
    return m


def generate_random_target(
    plant,
    bounds_np: np.ndarray,
    seed: int = 0,
    require_stable: bool = True,
    max_tries: int = 200,
    t_final: float = T_MAX,
) -> Tuple[Dict[str, float], Tuple[float, float]]:
    """
    Sample a random REACHABLE target by drawing (Kp, Ki) uniformly in `bounds_np`,
    simulating, and using the resulting metrics as the target profile.

    Because the target is the response of a real controller, U* = 0 is attainable.

    Parameters
    ----------
    bounds_np : np.ndarray, shape [2, 2]
        [[Kp_lo, Ki_lo], [Kp_hi, Ki_hi]].
    require_stable : bool
        If True, keep sampling until a "sensible" controller is found
        (MSE < 10 and settling < t_final), so the target is a good goal, not noise.

    Returns
    -------
    target_metrics : dict
    target_params  : (Kp, Ki)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds_np[0], bounds_np[1]

    best_fallback = None
    for _ in range(max_tries):
        kp, ki = rng.uniform(lo, hi)
        m = _eval_metrics(float(kp), float(ki), plant, t_final)
        stable = (m["tracking_mse"] < 10.0) and (m["settling_time"] < t_final - 1.0)
        if not require_stable or stable:
            return m, (float(kp), float(ki))
        if best_fallback is None or m["tracking_mse"] < best_fallback[0]["tracking_mse"]:
            best_fallback = (m, (float(kp), float(ki)))

    # No stable point found — return the least-bad sample.
    return best_fallback


def compute_ground_truth_optimum(
    plant,
    bounds_np: np.ndarray,
    utility: TargetSeekingUtility,
    n_grid: int = 30,
    t_final: float = T_MAX,
):
    """
    Dense-grid evaluation of `utility` for regret computation and the heatmap.

    Returns
    -------
    best_params : (Kp*, Ki*)
    U_star      : float          (max utility on the grid; ≈ 0 since target is reachable)
    grid_data   : dict with 'Kp', 'Ki', 'utility' flat arrays for plotting
    """
    kp_vals = np.linspace(bounds_np[0, 0], bounds_np[1, 0], n_grid)
    ki_vals = np.linspace(bounds_np[0, 1], bounds_np[1, 1], n_grid)

    best_u, best_params = -np.inf, (float(kp_vals[0]), float(ki_vals[0]))
    g_kp, g_ki, g_u = [], [], []

    for kp in kp_vals:
        for ki in ki_vals:
            m = _eval_metrics(float(kp), float(ki), plant, t_final)
            u = utility.utility(m)
            g_kp.append(kp); g_ki.append(ki); g_u.append(u)
            if u > best_u:
                best_u, best_params = u, (float(kp), float(ki))

    grid_data = {"Kp": np.array(g_kp), "Ki": np.array(g_ki), "utility": np.array(g_u)}
    return best_params, float(best_u), grid_data
