"""Visualization for trajectory optimization results.

Reproduces the plots from matlab/trajectory/ResultPlot.m using matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .trajectory_optimization import TrajectoryResult


def plot_trajectory_result(
    result: TrajectoryResult,
    save_dir: str | Path = "results",
    prefix: str = "trajopt",
) -> list[Path]:
    """Plot trajectory optimization results.

    Reproduces the three MATLAB figures:
      1. V-theta phase portrait + time histories (theta, V, h, q)
      2. Control inputs (dt, de, us)
      3. Angle of attack

    Args:
        result: TrajectoryResult from the optimizer.
        save_dir: Directory to save figures.
        prefix: Filename prefix for saved figures.

    Returns:
        List of saved figure paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t = result.time
    state = result.state
    control = result.control

    u = state[:, 0]
    w = state[:, 1]
    theta = state[:, 2]
    q = state[:, 3]
    h = state[:, 4]

    dt_ctrl = state[:, 8]
    de_ctrl = state[:, 9]
    us_ctrl = state[:, 10]

    V = np.sqrt(u**2 + w**2)
    theta_deg = np.degrees(theta)
    q_deg = np.degrees(q)
    alpha_deg = np.degrees(np.arcsin(np.clip(w / np.maximum(V, 1e-10), -1, 1)))

    saved = []

    # ---- Figure 1: Phase portrait + time histories ----
    fig1, axes = plt.subplots(4, 2, figsize=(10, 12))

    # Phase portrait (spans left column)
    ax_phase = plt.subplot(4, 2, (1, 7))
    ax_phase.plot(V, theta_deg, linewidth=1)
    ax_phase.set_xlabel(r"$V\;[m/s]$", fontsize=12)
    ax_phase.set_ylabel(r"$\theta\;[^\circ]$", fontsize=12)
    ax_phase.grid(True, alpha=0.3)

    # Time histories (right column)
    axes[0, 1].plot(t, theta_deg, linewidth=1)
    axes[0, 1].set_ylabel(r"$\theta\;[^\circ]$", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(t, V, linewidth=1)
    axes[1, 1].set_ylabel(r"$V\;[m/s]$", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(t, h, linewidth=1)
    axes[2, 1].set_ylabel(r"$h\;[m]$", fontsize=12)
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 1].plot(t, q_deg, linewidth=1)
    axes[3, 1].set_ylabel(r"$q\;[^\circ/s]$", fontsize=12)
    axes[3, 1].set_xlabel(r"$Time\;[s]$", fontsize=12)
    axes[3, 1].grid(True, alpha=0.3)

    # Hide unused left-column axes
    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)
    axes[3, 0].set_visible(False)

    fig1.suptitle("State Trajectories", fontsize=14)
    fig1.tight_layout()
    p1 = save_dir / f"{prefix}_states.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    saved.append(p1)

    # ---- Figure 2: Control inputs ----
    fig2, (ax_dt, ax_de, ax_us) = plt.subplots(3, 1, figsize=(8, 8))

    ax_dt.plot(t, dt_ctrl, linewidth=1, color="tab:blue")
    ax_dt.set_ylabel(r"$\delta_t$", fontsize=12)
    ax_dt.grid(True, alpha=0.3)

    ax_de.plot(t, de_ctrl, linewidth=1, color="tab:blue")
    ax_de.set_ylabel(r"$\delta_e\;[^\circ]$", fontsize=12)
    ax_de.grid(True, alpha=0.3)

    ax_us.plot(t, us_ctrl, linewidth=1, color="tab:blue")
    ax_us.set_ylabel(r"$u_s$", fontsize=12)
    ax_us.set_xlabel(r"$Time\;[s]$", fontsize=12)
    ax_us.grid(True, alpha=0.3)

    fig2.suptitle("Control Inputs", fontsize=14)
    fig2.tight_layout()
    p2 = save_dir / f"{prefix}_controls.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    saved.append(p2)

    # ---- Figure 3: Angle of attack ----
    fig3, ax_alpha = plt.subplots(figsize=(8, 4))
    ax_alpha.plot(t, alpha_deg, linewidth=1, color="tab:blue")
    ax_alpha.set_ylabel(r"$\alpha\;[^\circ]$", fontsize=12)
    ax_alpha.set_xlabel(r"$Time\;[s]$", fontsize=12)
    ax_alpha.grid(True, alpha=0.3)
    fig3.suptitle("Angle of Attack", fontsize=14)
    fig3.tight_layout()
    p3 = save_dir / f"{prefix}_alpha.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    saved.append(p3)

    return saved
