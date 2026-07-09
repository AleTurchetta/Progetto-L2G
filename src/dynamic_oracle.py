"""
dynamic_oracle.py
=================
A LATE SWITCHER that GENUINELY changes its mind.

Why base_oracle.LateSwitcherOracle is unsuitable for the validator study:
it inverts the label while regret stays measured on a FIXED utility, so a
validator that (correctly) confirms the switched preference makes regret
worse by construction — no method can win.

Why a weight-flip (s7-style) is also unsuitable: with TargetSeekingUtility
the target is REACHABLE, so U = 0 at the target for ANY weights — flipping
weights reshapes the landscape but never moves the argmax.  The human hasn't
actually changed what they want.

This oracle therefore holds TWO TargetSeekingUtility instances with two
DIFFERENT reachable targets.  Before `switch_iter` it answers honestly w.r.t.
utility 1; from `switch_iter` on, honestly w.r.t. utility 2.  Regret must be
measured against the CURRENTLY ACTIVE utility (dynamic regret):

    SR_t = U*_active(t) - U_active(t)(x_t^best)

with U*_1, U*_2 each computed once by grid search (both ~ 0, so the regret
scale is comparable before and after the switch).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from base_oracle import (
    BaseOracle,
    Persona,
    TargetSeekingUtility,
    generate_random_target,
    normalize_metrics,
)


class GenuineLateSwitcherOracle(BaseOracle):
    """
    Honest oracle whose hidden utility switches target at `switch_iter`.

    Consistent within each regime — the pre/post contradiction is real
    information, not noise.  A validator retest after the switch will
    CONFIRM the new preference (deterministically), which is the desired
    behaviour: the GP should follow the human's new goal.
    """

    persona = Persona.LATE_SWITCHER          # same label so plots line up

    def __init__(
        self,
        utility_pre: TargetSeekingUtility,
        utility_post: TargetSeekingUtility,
        seed: int = 42,
        verbose: bool = False,
        switch_iter: int = 8,
    ):
        super().__init__(utility_pre, seed=seed, verbose=verbose)
        self._u_pre = utility_pre
        self._u_post = utility_post
        self.switch_iter = int(switch_iter)

    # -- active utility ---------------------------------------------------------

    def active_utility(self, iteration: int) -> TargetSeekingUtility:
        return self._u_post if iteration >= self.switch_iter else self._u_pre

    def utility_at(self, metrics: Dict[str, float], iteration: int) -> float:
        """Utility under the regime ACTIVE at `iteration` (for dynamic regret)."""
        return self.active_utility(iteration).utility(metrics)

    # -- feedback ----------------------------------------------------------------

    def query(self, mA: dict, mB: dict, iteration: int = 1) -> dict:
        # Swap the working utility so _honest_output (preference + directional
        # constraints toward the ACTIVE target) reflects the current regime.
        self.utility_fn = self.active_utility(iteration)
        out = self._honest_output(mA, mB, iteration)
        self._finalise(out, mA, mB, iteration)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# second-target generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_distinct_target(
    plant,
    bounds_np: np.ndarray,
    ref_target: Dict[str, float],
    seed_start: int = 456,
    min_dist: float = 0.25,
    max_seeds: int = 60,
) -> Tuple[Dict[str, float], Tuple[float, float]]:
    """
    Sample a second REACHABLE target sufficiently far from `ref_target` in
    normalised metric space (so the switch is visible on the regret plot).

    Returns (target_metrics, (Kp, Ki)).  Falls back to the farthest sample
    found if none clears `min_dist`.
    """
    ref_n = normalize_metrics(ref_target)
    best = None            # (dist, target, params)
    for s in range(seed_start, seed_start + max_seeds):
        t, params = generate_random_target(plant, bounds_np, seed=s)
        d = float(np.linalg.norm(normalize_metrics(t) - ref_n))
        if best is None or d > best[0]:
            best = (d, t, params)
        if d >= min_dist:
            return t, params
    print(f"  [dynamic_oracle] WARNING: no target with dist>={min_dist} found; "
          f"using farthest (d={best[0]:.3f}).")
    return best[1], best[2]
