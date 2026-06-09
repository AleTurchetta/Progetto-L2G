from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

DEFAULT_EPS: float = 0.02


def convergence_iteration(
    regrets: Sequence[float],
    eps: float = DEFAULT_EPS,
    persistent: bool = True,
) -> Optional[int]:
    """
    First (1-indexed) iteration at which Simple Regret drops below epsilon(eps).

    Parameters
    ----------
    regrets : 1-D sequence
        Simple regret per iteration, SR_t = U* - U(x_best_t)  (already a utility
        gap; non-negative and bounded with the standardized utility).
    eps : float
        Absolute utility-gap threshold.  Default 0.02.
    persistent : bool
        If True, require the regret to stay below `eps` for all later iterations
        (a stricter, more honest definition that ignores a lucky one-off dip).
        If False, return the first crossing.

    Returns
    -------
    int | None
        The 1-indexed iteration, or None if convergence is never reached.
    """
    
    r = np.asarray(regrets, dtype=float).ravel()
    if r.size == 0:
        return None
    below = r < eps
    if not below.any():
        return None
    if not persistent:
        return int(np.argmax(below)) + 1
    # First index from which all subsequent values stay below eps.
    for t in range(r.size):
        if below[t:].all():
            return t + 1
    return None


def convergence_iteration_mean(
    regret_curves: np.ndarray,
    eps: float = DEFAULT_EPS,
    persistent: bool = True,
) -> Optional[int]:
    """
    Convergence iteration of the across-seed MEAN regret curve.

    Parameters
    ----------
    regret_curves : np.ndarray, shape [n_seeds, n_iters] (or [n_iters])
    eps, persistent : see `convergence_iteration`.
    """
    curves = np.atleast_2d(np.asarray(regret_curves, dtype=float))
    return convergence_iteration(curves.mean(axis=0), eps=eps, persistent=persistent)


def final_regret(regret_curves: np.ndarray) -> tuple[float, float]:
    """Mean ± std of the last-iteration regret across seeds."""
    curves = np.atleast_2d(np.asarray(regret_curves, dtype=float))
    last = curves[:, -1]
    return float(last.mean()), float(last.std())


if __name__ == "__main__":
    # quick sanity checks
    assert convergence_iteration([0.5, 0.3, 0.1, 0.01, 0.005]) == 4
    assert convergence_iteration([0.5, 0.01, 0.3, 0.005], persistent=True) == 4
    assert convergence_iteration([0.5, 0.01, 0.3, 0.005], persistent=False) == 2
    assert convergence_iteration([0.5, 0.3, 0.2]) is None
    # mean = [0.50, 0.03, 0.01]; only iter 3 is < eps=0.02
    assert convergence_iteration_mean(np.array([[0.4, 0.02, 0.01],
                                                [0.6, 0.04, 0.01]])) == 3
    print("convergence_utils self-test ✓")
