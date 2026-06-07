import numpy as np
import matplotlib.pyplot as plt
import control

# ============================================================
# 1) Mondo controllo: plant, PI, simulazione, metriche
# ============================================================

def make_plant(wn: float = 1.0, zeta: float = 0.2) -> control.TransferFunction:
    """
    Second-order plant:
        G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    """
    num = [wn ** 2]
    den = [1.0, 2.0 * zeta * wn, wn ** 2]
    return control.tf(num, den)


def make_pi(Kp: float, Ki: float) -> control.TransferFunction:
    """
    PI controller:
        C(s) = Kp + Ki/s = (Kp*s + Ki) / s
    """
    return control.tf([Kp, Ki], [1.0, 0.0])


def closed_loop_sys(Kp: float, Ki: float, plant: control.TransferFunction):
    """
    Closed-loop with unity feedback.
    """
    C = make_pi(Kp, Ki)
    L = C * plant
    T = control.feedback(L, 1)  # unity feedback
    return T


def simulate_step(Kp: float, Ki: float, plant, t_final: float = 40.0, n_points: int = 1000):
    """
    Simulate closed-loop step response.
    """
    T = np.linspace(0, t_final, n_points)
    sys_cl = closed_loop_sys(Kp, Ki, plant)
    t, y = control.step_response(sys_cl, T)
    return t, np.squeeze(y)


def compute_metrics(t: np.ndarray, y: np.ndarray, ref: float = 1.0, settle_tol: float = 0.02):
    """
    Interpretable metrics:
      - tracking_mse
      - overshoot_pct
      - settling_time
    """
    y_final = ref
    e = ref - y

    # MSE (integrale / durata)
    tracking_mse = np.trapezoid(e ** 2, t) / (t[-1] - t[0])

    # Overshoot (% rispetto a ref)
    peak = np.max(y)
    overshoot_pct = max(0.0, (peak - y_final) / y_final * 100.0)

    # Settling time: primo istante da cui rimane entro ±2%
    within = np.abs(y - y_final) <= settle_tol * abs(y_final)
    settling_time = np.nan
    if np.any(within):
        idxs = np.where(within)[0]
        for idx in idxs:
            if np.all(within[idx:]):
                settling_time = t[idx]
                break

    return {
    "tracking_mse": float(tracking_mse),
    "overshoot_pct": float(overshoot_pct),
    "settling_time": float(settling_time) if not np.isnan(settling_time) else float("nan"),
}


def plot_two_responses(
    tA, yA, metricsA, thetaA,
    tB, yB, metricsB, thetaB,
    ref=1.0,
):
    """
    Plot side-by-side A vs B for pairwise comparison.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # A
    ax[0].plot(tA, np.ones_like(tA) * ref, "--", label="ref")
    ax[0].plot(tA, yA, label="y_A(t)")
    ax[0].set_title(f"A: Kp={thetaA[0]:.3f}, Ki={thetaA[1]:.3f}")
    ax[0].grid(True)
    ax[0].legend()
    txtA = (
        f"MSE={metricsA['tracking_mse']:.3g}\n"
        f"Overshoot={metricsA['overshoot_pct']:.1f}%\n"
        f"Ts={metricsA['settling_time']:.3g} s"
    )
    ax[0].text(0.02, 0.02, txtA, transform=ax[0].transAxes,
               bbox=dict(boxstyle="round", alpha=0.3), fontsize=8)

    # B
    ax[1].plot(tB, np.ones_like(tB) * ref, "--", label="ref")
    ax[1].plot(tB, yB, label="y_B(t)")
    ax[1].set_title(f"B: Kp={thetaB[0]:.3f}, Ki={thetaB[1]:.3f}")
    ax[1].grid(True)
    ax[1].legend()
    txtB = (
        f"MSE={metricsB['tracking_mse']:.3g}\n"
        f"Overshoot={metricsB['overshoot_pct']:.1f}%\n"
        f"Ts={metricsB['settling_time']:.3g} s"
    )
    ax[1].text(0.02, 0.02, txtB, transform=ax[1].transAxes,
               bbox=dict(boxstyle="round", alpha=0.3), fontsize=8)

    fig.suptitle("Pairwise comparison: A vs B")
    plt.tight_layout()
    plt.show()