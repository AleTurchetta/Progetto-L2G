"""
oracle.py
=========
Deterministic pseudo-human oracle for L2G stress-testing.

Replaces the LLM + real human interaction with a reproducible agent that:
1. Has a declared "hidden utility" over the metric space.
2. Produces valid L2GOutput JSON every iteration given A/B metrics.
3. Supports multiple "personas" (stress scenarios).
4. Has configurable noise to simulate human inconsistency.

Usage
-----
    from oracle import Oracle, OraclePersona

    oracle = Oracle(
        persona=OraclePersona.MONOTONE,
        noise_level=0.0,
        seed=42,
    )

    # Each iteration, give it the real metrics from the simulation:
    output_dict = oracle.query(metrics_A, metrics_B, iteration=1)

    # Feed directly into L2GEngine.process_feedback():
    l2g.process_feedback(candidates_dict, feedback_text="", simulated_output=output_dict)

    # At the end, access the oracle's utility for regret computation:
    u_A = oracle.utility(metrics_A)
    u_B = oracle.utility(metrics_B)
"""

from __future__ import annotations

import random
import numpy as np
from enum import Enum, auto
from typing import Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ORACLE PERSONAS  (stress scenarios)
# ─────────────────────────────────────────────────────────────────────────────

class OraclePersona(Enum):
    MONOTONE             = auto()   # S1: always prefers lower MSE, no constraints
    MONOTONE_CONSTRAINED = auto()   # S1b: MSE-dominant preference + aligned relative constraint
    NOISY                = auto()   # S2: 20% random flip in preference label
    CONTRADICTORY        = auto()   # S3: prefers B but adds constraint B already violates
    DRASTIC_ABSOLUTE     = auto()   # S4: hard absolute threshold ("no overshoot" → < 1%)
    DRASTIC_RELATIVE     = auto()   # S5: "better than A in everything"
    DIRECTION_ONLY       = auto()   # S6: never reveals preference, only adds direction constraints
    LATE_SWITCHER        = auto()   # S7: consistent for 3 iters, then reverses all weights
    AMBIGUOUS_POSITIVE   = auto()   # S8: always says "I prefer B" even when B is worse


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TARGET PROFILES  (hidden utility parameters per persona)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetProfile:
    """
    Defines the oracle's hidden utility function:
        U(m) = - w_ov*overshoot_pct/100 - w_ts*settling_time/T_max - w_mse*log(mse+1)

    Plus optional hard thresholds that will be converted to constraints.
    """
    # Utility weights (non-negative, will be L1-normalised internally)
    w_overshoot:    float = 1.0
    w_settling:     float = 1.0
    w_mse:          float = 1.0

    # Hard thresholds (None means no constraint generated for that metric)
    max_overshoot:  Optional[float] = None   # e.g., 5.0 → "overshoot < 5%"
    max_settling:   Optional[float] = None   # e.g., 10.0 → "settling < 10 s"
    max_mse:        Optional[float] = None   # e.g., 0.5 → "MSE < 0.5"

    # Normalisation constant for settling time
    T_max: float = 40.0

    # Constraint margin (tolerance around hard thresholds)
    margin: float = 0.0


# Pre-defined profiles for each persona
_PROFILES: Dict[OraclePersona, TargetProfile] = {
    OraclePersona.MONOTONE: TargetProfile(
        w_overshoot=0.1, w_settling=0.1, w_mse=1.0,
    ),
    OraclePersona.MONOTONE_CONSTRAINED: TargetProfile(
        w_overshoot=0.1, w_settling=0.1, w_mse=1.0,
    ),  # identical utility to MONOTONE; constraint is relative, built in handler

    OraclePersona.NOISY: TargetProfile(
        w_overshoot=0.5, w_settling=0.5, w_mse=1.0,
    ),
    OraclePersona.CONTRADICTORY: TargetProfile(
        w_overshoot=1.0, w_settling=1.0, w_mse=1.0,
        max_overshoot=3.0,   # tight absolute constraint
    ),
    OraclePersona.DRASTIC_ABSOLUTE: TargetProfile(
        w_overshoot=2.0, w_settling=0.5, w_mse=0.5,
        max_overshoot=1.0,   # "virtually no overshoot"
        max_settling=15.0,
    ),
    OraclePersona.DRASTIC_RELATIVE: TargetProfile(
        w_overshoot=1.0, w_settling=1.0, w_mse=1.0,
        # No absolute thresholds; constraints generated as "better than A"
    ),
    OraclePersona.DIRECTION_ONLY: TargetProfile(
        w_overshoot=1.5, w_settling=0.5, w_mse=1.0,
        max_overshoot=5.0,
    ),
    OraclePersona.LATE_SWITCHER: TargetProfile(
        # Initial weights (will flip at iteration late_switch_iter)
        w_overshoot=2.0, w_settling=0.2, w_mse=0.5,
    ),
    OraclePersona.AMBIGUOUS_POSITIVE: TargetProfile(
        w_overshoot=1.0, w_settling=1.0, w_mse=1.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ORACLE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Oracle:
    """
    Deterministic pseudo-human oracle.

    Parameters
    ----------
    persona : OraclePersona
        Stress scenario to simulate.
    noise_level : float
        Probability of randomly flipping the preference label (0.0 = deterministic).
    seed : int
        Random seed for reproducibility.
    late_switch_iter : int
        For LATE_SWITCHER: iteration at which weights reverse.
    verbose : bool
        Print human-readable explanation of each output.
    """

    def __init__(
        self,
        persona: OraclePersona = OraclePersona.MONOTONE,
        noise_level: float = 0.0,
        seed: int = 42,
        late_switch_iter: int = 4,
        verbose: bool = True,
    ):
        self.persona = persona
        self.noise_level = noise_level
        self.seed = seed
        self.late_switch_iter = late_switch_iter
        self.verbose = verbose

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Copy the profile so we can mutate it for LATE_SWITCHER
        self._profile: TargetProfile = TargetProfile(**vars(_PROFILES[persona]))

        # History of outputs (useful for analysis)
        self.history: list[dict] = []

    # ─── Public API ──────────────────────────────────────────────────────────

    def utility(self, metrics: Dict[str, float]) -> float:
        """
        Compute the oracle's hidden utility for a metrics dict.
        Higher is better (utility is negated cost).
        Note: always deterministic — noise only affects the preference label in query().
        """
        p = self._profile
        total_w = p.w_overshoot + p.w_settling + p.w_mse + 1e-9
        u = (
            - p.w_overshoot / total_w * metrics.get("overshoot_pct", 100.0) / 100.0
            - p.w_settling  / total_w * metrics.get("settling_time", p.T_max) / p.T_max
            - p.w_mse       / total_w * np.log1p(metrics.get("tracking_mse", 1e5))
        )
        return float(u)

    def query(
        self,
        metrics_A: Dict[str, float],
        metrics_B: Dict[str, float],
        iteration: int = 1,
    ) -> dict:
        """
        Main entry point. Returns a dict that can be passed as
        `simulated_output` to L2GEngine.process_feedback().

        Parameters
        ----------
        metrics_A : dict
            Real simulated metrics for candidate A (current best / baseline).
        metrics_B : dict
            Real simulated metrics for candidate B (newly proposed).
        iteration : int
            Current iteration number (1-indexed).

        Returns
        -------
        dict
            A dict matching the L2GOutput schema:
            {
                "feedback_type": "both"|"preference_only"|"direction_only"|"none",
                "preference": {"preferred_candidate": ..., "other_candidate": ..., "confidence": ...} | None,
                "constraints": [...]
            }
        """
        # Possible late switch of weights
        if self.persona == OraclePersona.LATE_SWITCHER and iteration >= self.late_switch_iter:
            self._flip_weights()

        # Route to persona handler
        handler = {
            OraclePersona.MONOTONE:             self._handle_monotone,
            OraclePersona.MONOTONE_CONSTRAINED: self._handle_monotone_constrained,
            OraclePersona.NOISY:                self._handle_noisy,
            OraclePersona.CONTRADICTORY:        self._handle_contradictory,
            OraclePersona.DRASTIC_ABSOLUTE:     self._handle_drastic_absolute,
            OraclePersona.DRASTIC_RELATIVE:     self._handle_drastic_relative,
            OraclePersona.DIRECTION_ONLY:       self._handle_direction_only,
            OraclePersona.LATE_SWITCHER:        self._handle_late_switcher,
            OraclePersona.AMBIGUOUS_POSITIVE:   self._handle_ambiguous_positive,
        }[self.persona]

        output = handler(metrics_A, metrics_B, iteration)

        # Apply global noise (random preference flip)
        if self.noise_level > 0 and output.get("preference"):
            if self._rng.random() < self.noise_level:
                pref = output["preference"]
                pref["preferred_candidate"], pref["other_candidate"] = (
                    pref["other_candidate"], pref["preferred_candidate"]
                )
                pref["confidence"] = max(0.1, pref["confidence"] - 0.2)
                if self.verbose:
                    print(f"  [Oracle] NOISE FLIP applied (p={self.noise_level:.0%})")

        self.history.append({"iteration": iteration, "output": output})

        if self.verbose:
            self._print_output(output, metrics_A, metrics_B)

        return output

    # ─── Persona Handlers ────────────────────────────────────────────────────

    def _handle_monotone(self, mA, mB, iteration):
        """S1: Pure utility comparison, no constraints."""
        u_A = self.utility(mA)
        u_B = self.utility(mB)
        winner, loser = ("B", "A") if u_B > u_A else ("A", "B")
        confidence = min(1.0, abs(u_B - u_A) * 5 + 0.6)

        return {
            "feedback_type": "preference_only",
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": round(confidence, 2),
            },
            "constraints": [],
        }

    def _handle_monotone_constrained(self, mA, mB, iteration):
        """S1b: Utility comparison + relative constraint (no MSE regression vs A)."""
        u_A = self.utility(mA)
        u_B = self.utility(mB)
        winner, loser = ("B", "A") if u_B > u_A else ("A", "B")
        confidence = min(1.0, abs(u_B - u_A) * 5 + 0.6)

        # Relative constraint: next candidate must not be worse than A on MSE
        constraints = [{
            "id": f"mse_no_regress_iter{iteration}",
            "subfunction_id": "tracking_mse",
            "constraint_type": "directional_improvement",
            "operator": "<=",
            "reference_candidate": "A",
            "threshold": None,
            "margin": 0.05,   # 5% tolerance so it's not razor-thin
            "weight": 1.0,
            "confidence": 0.9,
        }]

        return {
            "feedback_type": "both",
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": round(confidence, 2),
            },
            "constraints": constraints,
        }

    def _handle_noisy(self, mA, mB, iteration):
        """S2: Utility comparison with built-in noise (handled in query())."""
        return self._handle_monotone(mA, mB, iteration)

    def _handle_contradictory(self, mA, mB, iteration):
        """
        S3: Prefers B (by utility), but then adds a hard constraint that B
        already violates — forcing the system into an infeasible state.
        """
        # Force preference toward B (even if B is worse)
        winner, loser = "B", "A"
        confidence = 0.75

        # Build a constraint that B *already* violates (max_overshoot tighter than B's actual value)
        b_overshoot = mB.get("overshoot_pct", 50.0)
        # Set threshold slightly below B's actual value → contradiction
        tight_threshold = max(0.1, b_overshoot * 0.7)

        constraints = [
            {
                "id": "contradictory_overshoot",
                "subfunction_id": "overshoot_pct",
                "constraint_type": "upper_bound_absolute",
                "operator": "<=",
                "threshold": round(tight_threshold, 2),
                "margin": 0.0,
                "weight": 1.0,
                "confidence": 0.9,
            }
        ]

        return {
            "feedback_type": "both",
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": confidence,
            },
            "constraints": constraints,
        }

    def _handle_drastic_absolute(self, mA, mB, iteration):
        """
        S4: Hard absolute constraint (virtually no overshoot, fast settling).
        Also expresses a preference based on utility.
        """
        u_A = self.utility(mA)
        u_B = self.utility(mB)
        winner, loser = ("B", "A") if u_B > u_A else ("A", "B")
        confidence = 0.9

        p = self._profile
        constraints = []

        if p.max_overshoot is not None:
            constraints.append({
                "id": f"abs_overshoot_iter{iteration}",
                "subfunction_id": "overshoot_pct",
                "constraint_type": "upper_bound_absolute",
                "operator": "<=",
                "threshold": p.max_overshoot,
                "margin": 0.0,
                "weight": 1.5,
                "confidence": 0.95,
            })

        if p.max_settling is not None:
            constraints.append({
                "id": f"abs_settling_iter{iteration}",
                "subfunction_id": "settling_time",
                "constraint_type": "upper_bound_absolute",
                "operator": "<=",
                "threshold": p.max_settling,
                "margin": 0.0,
                "weight": 1.0,
                "confidence": 0.9,
            })

        feedback_type = "both" if constraints else "preference_only"

        return {
            "feedback_type": feedback_type,
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": confidence,
            },
            "constraints": constraints,
        }

    def _handle_drastic_relative(self, mA, mB, iteration):
        """
        S5: "Better than A in everything" — relative constraints on all metrics.
        """
        u_A = self.utility(mA)
        u_B = self.utility(mB)
        winner, loser = ("B", "A") if u_B > u_A else ("A", "B")

        constraints = []
        for metric in ["overshoot_pct", "settling_time", "tracking_mse"]:
            constraints.append({
                "id": f"rel_{metric}_vs_A_iter{iteration}",
                "subfunction_id": metric,
                "constraint_type": "directional_improvement",
                "operator": "<=",
                "reference_candidate": "A",
                "threshold": None,
                "margin": 0.0,
                "weight": 1.0,
                "confidence": 0.85,
            })

        return {
            "feedback_type": "both",
            "preference": {
                "preferred_candidate": winner,
                "other_candidate": loser,
                "confidence": 0.8,
            },
            "constraints": constraints,
        }

    def _handle_direction_only(self, mA, mB, iteration):
        """
        S6: No preference label; only directional constraints.
        Tests the direction_only code path end-to-end.
        """
        p = self._profile
        constraints = []

        # Pick the worst metric and push it toward the profile target
        metrics_ranked = sorted(
            [
                ("overshoot_pct",  mB.get("overshoot_pct", 100.0),  p.max_overshoot or 5.0),
                ("settling_time",  mB.get("settling_time", 40.0),    p.max_settling  or 10.0),
                ("tracking_mse",   mB.get("tracking_mse", 1e4),      p.max_mse       or 0.5),
            ],
            key=lambda x: (x[1] - x[2]) / (abs(x[2]) + 1e-6),  # how far above target?
            reverse=True,
        )

        worst_metric, worst_val, target = metrics_ranked[0]

        if worst_val > target:
            constraints.append({
                "id": f"direction_{worst_metric}_iter{iteration}",
                "subfunction_id": worst_metric,
                "constraint_type": "upper_bound_absolute",
                "operator": "<=",
                "threshold": round(target, 3),
                "margin": 0.0,
                "weight": 1.2,
                "confidence": 0.9,
            })

        return {
            "feedback_type": "direction_only" if constraints else "none",
            "preference": None,
            "constraints": constraints,
        }

    def _handle_late_switcher(self, mA, mB, iteration):
        """
        S7: Consistent preference for first `late_switch_iter` iterations,
        then weights flip (handled by _flip_weights in query()).
        """
        return self._handle_monotone(mA, mB, iteration)

    def _handle_ambiguous_positive(self, mA, mB, iteration):
        """
        S8: Always says "I prefer B" regardless of actual quality.
        Tests whether the system exploits false positive feedback.
        """
        return {
            "feedback_type": "preference_only",
            "preference": {
                "preferred_candidate": "B",
                "other_candidate": "A",
                "confidence": 0.9,
            },
            "constraints": [],
        }

    # ─── Helpers ────────────────────────────────────────────────────

    def _flip_weights(self):
        """Called once when LATE_SWITCHER reaches the switch iteration."""
        p = self._profile
        # Swap overshoot and MSE weights, double settling weight
        p.w_overshoot, p.w_mse = p.w_mse, p.w_overshoot
        p.w_settling = p.w_settling * 2.0
        if self.verbose:
            print(f"  [Oracle] LATE SWITCH: weights flipped to "
                  f"ov={p.w_overshoot:.1f}, ts={p.w_settling:.1f}, mse={p.w_mse:.1f}")
        # Only flip once
        self._flip_weights = lambda: None  # noqa: disable further calls

    def _print_output(self, output, mA, mB):
        pref = output.get("preference")
        winner = pref["preferred_candidate"] if pref else "—"
        n_constraints = len(output.get("constraints", []))
        u_A = self.utility(mA)
        u_B = self.utility(mB)
        print(
            f"  [Oracle|{self.persona.name}] "
            f"type={output['feedback_type']} | "
            f"winner={winner} | "
            f"U(A)={u_A:.4f} U(B)={u_B:.4f} | "
            f"constraints={n_constraints}"
        )
        for c in output.get("constraints", []):
            thr = c.get("threshold", "relative")
            print(f"    └─ {c['subfunction_id']} {c['operator']} {thr}  (type={c['constraint_type']})")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPERIMENT RUNNER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ground_truth_optimum(plant, bounds_np, oracle: "Oracle", n_grid: int = 50, t_final: float = 40.0):
    """
    Evaluates the oracle's utility on a dense grid to find the ground-truth optimum.
    Use once at the start of each experiment for regret calculation.

    The utility is evaluated under the provided oracle's hidden utility function,
    so the ground truth is always consistent with the persona being tested.

    Parameters
    ----------
    plant : control system plant object
    bounds_np : np.ndarray, shape [2, 2]
        [[Kp_min, Ki_min], [Kp_max, Ki_max]]
    oracle : Oracle
        The experiment oracle instance. Its utility() method is deterministic
        (noise only affects preference labels in query(), not utility values).
    n_grid : int
        Grid resolution per axis. Total evaluations = n_grid².
    t_final : float
        Simulation horizon.

    Returns
    -------
    best_params : (Kp*, Ki*)
    best_utility : float
    grid_data : dict with 'Kp', 'Ki', 'utility' arrays for heatmap plotting
    """
    import control_utils as cu
    import torch

    kp_vals = np.linspace(bounds_np[0, 0], bounds_np[1, 0], n_grid)
    ki_vals = np.linspace(bounds_np[0, 1], bounds_np[1, 1], n_grid)

    best_u = -np.inf
    best_params = (kp_vals[0], ki_vals[0])

    grid_kp, grid_ki, grid_u = [], [], []

    total = n_grid * n_grid
    for idx, kp in enumerate(kp_vals):
        if idx % 5 == 0:
            print(f"  Grid search: {idx * n_grid}/{total}...", end="\r")
        for ki in ki_vals:
            x = torch.tensor([[kp, ki]], dtype=torch.double)
            t, y, m = _eval(x, plant, t_final)
            u = oracle.utility(m)
            grid_kp.append(kp)
            grid_ki.append(ki)
            grid_u.append(u)
            if u > best_u:
                best_u = u
                best_params = (kp, ki)

    print(f"  Grid search: {total}/{total} — done.        ")

    return best_params, best_u, {
        "Kp": np.array(grid_kp),
        "Ki": np.array(grid_ki),
        "utility": np.array(grid_u),
    }


def _eval(x_tensor, plant, t_final=40.0):
    """Thin simulation wrapper (mirrors evaluate_candidate in run scripts)."""
    import control_utils as cu
    import numpy as np
    kp = float(x_tensor[0, 0].item())
    ki = float(x_tensor[0, 1].item())
    t, y = cu.simulate_step(kp, ki, plant, t_final=t_final)
    m = cu.compute_metrics(t, y)
    if np.isnan(m["settling_time"]):
        m["settling_time"] = t_final + 10.0
    if m["tracking_mse"] > 1e4 or np.isnan(m["tracking_mse"]):
        m["tracking_mse"] = 1e5
    return t, y, m


# ─────────────────────────────────────────────────────────────────────────────
# 5.  QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Oracle self-test ===\n")

    fake_mA = {"overshoot_pct": 25.0, "settling_time": 12.0, "tracking_mse": 0.8}
    fake_mB = {"overshoot_pct": 8.0,  "settling_time": 7.0,  "tracking_mse": 0.3}

    for persona in OraclePersona:
        print(f"\n--- Persona: {persona.name} ---")
        o = Oracle(persona=persona, noise_level=0.0, seed=42, verbose=True)
        out = o.query(fake_mA, fake_mB, iteration=1)
        assert "feedback_type" in out
        assert "preference" in out
        assert "constraints" in out
        print(f"  Output keys valid ✓")

    print("\n=== All personas passed self-test ===")
