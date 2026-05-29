"""Visualization for differential-flatness-based trajectory results.

Replaces matlab/analysis/controller/visualize_trajectory.m,
visualize_simulation_results.m, visualize_transition_corridor.m,
visualize_multiple_trim_results.m, and visualize_transition_strategy.m.

Uses the _sci_style() context manager and COLORS palette from plotting.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .plotting import _sci_style, COLORS
from .df_trim import DFTrimMesh
from .df_trajectory_optimization import DFTrajectoryResult
from .baseline_trajectory import BaselineTrajectoryResult
from .trim import R2D


# ---------------------------------------------------------------------------
# DF trajectory visualization (3x3 subplot)
# ---------------------------------------------------------------------------

def plot_df_trajectory(
    result: DFTrajectoryResult,
    save_dir: str | Path,
    filename: str = "df_trajectory",
) -> list[Path]:
    """Plot DF trajectory optimization result (3x3 subplot grid).

    Ported from matlab/analysis/controller/visualize_trajectory.m.

    Args:
        result: DF trajectory optimization result
        save_dir: Directory to save plots
        filename: Base filename (without extension)

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    t = result.time
    gamma_deg = np.degrees(result.gamma)
    alpha_deg = np.degrees(result.alpha)
    theta_deg = np.degrees(result.theta)
    q_deg = np.degrees(result.q)
    x_pos = result.V * np.cos(result.gamma) * t

    with _sci_style():
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))

        # (a) Flight trajectory
        ax = axes[0, 0]
        ax.plot(x_pos, result.h, color=COLORS["blue"], lw=2)
        ax.plot(x_pos[0], result.h[0], 'o', color=COLORS["teal"], ms=6, zorder=5)
        ax.plot(x_pos[-1], result.h[-1], 'o', color=COLORS["red"], ms=6, zorder=5)
        ax.set(xlabel=r"$x$ [m]", ylabel=r"$h$ [m]", title="(a) Flight Trajectory")
        ax.legend(["Trajectory", "Start", "End"], fontsize=8)
        ax.set_axisbelow(True)

        # (b) Velocity
        ax = axes[0, 1]
        ax.plot(t, result.V, color=COLORS["red"], lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$V$ [m/s]", title="(b) Velocity")
        ax.set_axisbelow(True)

        # (c) Flight path angle
        ax = axes[0, 2]
        ax.plot(t, gamma_deg, color="#9C27B0", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\gamma$ [deg]", title="(c) Flight Path Angle")
        ax.set_axisbelow(True)

        # (d) Alpha and theta
        ax = axes[1, 0]
        ax.plot(t, alpha_deg, color=COLORS["blue"], lw=2, label=r"$\alpha$")
        ax.plot(t, theta_deg, color=COLORS["red"], lw=2, ls="--", label=r"$\theta$")
        ax.set(xlabel="Time [s]", ylabel="Angle [deg]", title="(d) AoA and Pitch Angle")
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

        # (e) Acceleration
        ax = axes[1, 1]
        vdot = np.gradient(result.V, t)
        ax.plot(t, vdot, color=COLORS["teal"], lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\dot{V}$ [m/s$^2$]", title="(e) Acceleration")
        ax.set_axisbelow(True)

        # (f) FPA rate
        ax = axes[1, 2]
        gamma_rate = np.gradient(gamma_deg, t)
        ax.plot(t, gamma_rate, color="#00BCD4", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\dot{\gamma}$ [deg/s]", title="(f) FPA Rate")
        ax.set_axisbelow(True)

        # (g) Pitch rate
        ax = axes[2, 0]
        ax.plot(t, q_deg, color="#212121", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$q$ [deg/s]", title="(g) Pitch Rate")
        ax.set_axisbelow(True)

        # (h) Throttle
        ax = axes[2, 1]
        ax.plot(t, result.throttle, color=COLORS["orange"], lw=2)
        ax.set(xlabel="Time [s]", ylabel="Throttle [N]", title="(h) Throttle Input")
        ax.set_axisbelow(True)

        # (i) Elevator
        ax = axes[2, 2]
        ax.plot(t, result.elevator, color="#7B1FA2", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\delta_e$ [deg]", title="(i) Elevator")
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Simulation comparison (actual vs reference)
# ---------------------------------------------------------------------------

def plot_df_simulation_comparison(
    sim_data: np.ndarray,
    save_dir: str | Path,
    filename: str = "sim_comparison",
) -> list[Path]:
    """Plot simulation comparison: actual vs reference states.

    Ported from matlab/analysis/controller/visualize_simulation_results.m.

    sim_data columns:
        0: time, 1: ref_V, 2: ref_gamma, 3: ref_alpha, 4: ref_theta, 5: ref_q,
        6: V, 7: alpha, 8: theta, 9: q, 10: gamma, 11: h, 12: throttle, 13: elevator

    Args:
        sim_data: 2D array with simulation data
        save_dir: Directory to save plots
        filename: Base filename

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    t = sim_data[:, 0]
    ref_V, ref_gamma, ref_alpha, ref_theta, ref_q = sim_data[:, 1:6].T
    V, alpha, theta, q, gamma, h = sim_data[:, 6:12].T
    throttle, elevator = sim_data[:, 12:14].T

    with _sci_style():
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))

        # (a) Velocity
        ax = axes[0, 0]
        ax.plot(t, V, color=COLORS["blue"], lw=2, label="Actual")
        ax.plot(t, ref_V, color=COLORS["orange"], lw=1.5, ls="--", label="Reference")
        ax.set(xlabel="Time [s]", ylabel=r"$V$ [m/s]", title="(a) Velocity")
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

        # (b) Flight path angle
        ax = axes[0, 1]
        ax.plot(t, np.degrees(gamma), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_gamma), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\gamma$ [deg]", title="(b) Flight Path Angle")
        ax.set_axisbelow(True)

        # (c) Pitch angle
        ax = axes[0, 2]
        ax.plot(t, np.degrees(theta), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_theta), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\theta$ [deg]", title="(c) Pitch Angle")
        ax.set_axisbelow(True)

        # (d) Angle of attack
        ax = axes[1, 0]
        ax.plot(t, np.degrees(alpha), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_alpha), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\alpha$ [deg]", title="(d) Angle of Attack")
        ax.set_axisbelow(True)

        # (e) Pitch rate
        ax = axes[1, 1]
        ax.plot(t, np.degrees(q), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_q), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$q$ [deg/s]", title="(e) Pitch Rate")
        ax.set_axisbelow(True)

        # (f) V-gamma phase plane
        ax = axes[1, 2]
        ax.plot(V, np.degrees(gamma), color=COLORS["blue"], lw=2, label="Actual")
        ax.plot(ref_V, np.degrees(ref_gamma), color=COLORS["orange"], lw=1.5, ls="--", label="Reference")
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]", title=r"(f) V-$\gamma$ Phase")
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Transition corridor heatmap
# ---------------------------------------------------------------------------

def plot_df_corridor(
    mesh: DFTrimMesh,
    data_grid: np.ndarray,
    label: str,
    save_dir: str | Path,
    filename: Optional[str] = None,
    clim: Optional[tuple[float, float]] = None,
) -> list[Path]:
    """Plot a single heatmap over the DF transition corridor.

    Ported from matlab/analysis/controller/visualize_transition_corridor.m.

    Args:
        mesh: DF trim mesh result
        data_grid: 2D data array (same shape as mesh.V_grid)
        label: Colorbar label (e.g., r"$\alpha$ [deg]")
        save_dir: Directory to save plots
        filename: Filename (default: derived from label)
        clim: Color limits (min, max)

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if filename is None:
        filename = "df_corridor_" + label.replace("$", "").replace("\\", "").replace(" ", "_").replace("[", "").replace("]", "")

    # Apply feasibility mask
    data_plot = data_grid.copy()
    data_plot[mesh.flag_grid == 0] = np.nan

    with _sci_style():
        fig, ax = plt.subplots(figsize=(8, 6))

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='white')

        h = ax.pcolormesh(mesh.V_grid, mesh.gamma_grid, data_plot,
                          cmap=cmap, shading='auto')
        if clim is not None:
            h.set_clim(*clim)

        # Contour lines
        with np.errstate(invalid='ignore'):
            cs = ax.contour(mesh.V_grid, mesh.gamma_grid, data_plot,
                            colors='k', linewidths=0.5, alpha=0.5)
            ax.clabel(cs, fontsize=7)

        # Corridor boundary
        ax.contour(mesh.V_grid, mesh.gamma_grid, mesh.flag_grid,
                   levels=[0.5], colors='k', linewidths=2, linestyles='--')

        cb = fig.colorbar(h, ax=ax)
        cb.set_label(label)
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]")
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_df_corridor_suite(
    mesh: DFTrimMesh,
    save_dir: str | Path,
) -> list[Path]:
    """Plot full suite of corridor heatmaps (alpha, theta, q, T, ele).

    Args:
        mesh: DF trim mesh result
        save_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    saved = []
    configs = [
        (mesh.alpha_grid, r"$\alpha$ [deg]", "alpha"),
        (mesh.theta_grid, r"$\theta$ [deg]", "theta"),
        (mesh.q_grid, r"$q$ [deg/s]", "q"),
        (mesh.T_grid, r"$T$ [N]", "thrust"),
        (mesh.ele_grid, r"$\delta_e$ [deg]", "elevator"),
        (mesh.qmax_grid, r"$q_{max}$ [deg/s]", "qmax"),
    ]
    for grid, label, name in configs:
        saved.extend(plot_df_corridor(mesh, grid, label, save_dir, filename=f"corridor_{name}"))
    return saved


# ---------------------------------------------------------------------------
# Trajectory comparison on corridor
# ---------------------------------------------------------------------------

def plot_df_trajectory_comparison(
    proposed: DFTrajectoryResult,
    baseline: BaselineTrajectoryResult,
    mesh: DFTrimMesh,
    save_dir: str | Path,
    filename: str = "trajectory_comparison",
) -> list[Path]:
    """Plot proposed and baseline trajectories on the corridor feasibility map.

    Ported from matlab/analysis/controller/validate_trajectory + trajectory_compare_plot.

    Args:
        proposed: DF trajectory optimization result
        baseline: Baseline corridor path result
        mesh: DF trim mesh
        save_dir: Directory to save plots
        filename: Base filename

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Build pitch rate margin grid for background
    margin_grid = mesh.qmax_grid - np.abs(mesh.q_grid)
    margin_plot = margin_grid.copy()
    margin_plot[mesh.flag_grid == 0] = np.nan

    with _sci_style():
        fig, ax = plt.subplots(figsize=(9, 6))

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='white')

        h = ax.pcolormesh(mesh.V_grid, mesh.gamma_grid, margin_plot,
                          cmap=cmap, shading='auto')
        cb = fig.colorbar(h, ax=ax)
        cb.set_label(r"$q_{margin}$ [deg/s]")

        # Corridor boundary
        ax.contour(mesh.V_grid, mesh.gamma_grid, mesh.flag_grid,
                   levels=[0.5], colors='k', linewidths=2, linestyles='--',
                   label="Corridor boundary")

        # Baseline path
        if baseline.optimal_path.size > 0:
            ax.plot(baseline.optimal_path[:, 0], baseline.optimal_path[:, 1],
                    color=COLORS["orange"], lw=2, marker='s', ms=4,
                    label="Baseline path")

        # Proposed trajectory
        ax.plot(proposed.V, np.degrees(proposed.gamma),
                color=COLORS["blue"], lw=2.5, label="Proposed trajectory")

        # Start/end markers
        ax.plot(proposed.V[0], np.degrees(proposed.gamma[0]),
                'o', color=COLORS["teal"], ms=8, zorder=5, label="Start")
        ax.plot(proposed.V[-1], np.degrees(proposed.gamma[-1]),
                'o', color=COLORS["red"], ms=8, zorder=5, label="End")

        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]",
               title="Transition Trajectory Comparison")
        ax.legend(fontsize=8, loc="best")
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Multiple corridor comparison (3D surface + 2D heatmaps)
# ---------------------------------------------------------------------------

def plot_df_multiple_corridors(
    mesh_list: Sequence[DFTrimMesh],
    labels: Sequence[str],
    data_key: str,
    save_dir: str | Path,
    filename: Optional[str] = None,
) -> list[Path]:
    """Plot multiple corridor datasets for comparison.

    Ported from matlab/analysis/controller/visualize_multiple_trim_results.m.

    Args:
        mesh_list: List of DF trim meshes
        labels: Labels for each dataset
        data_key: Which grid to plot ('alpha', 'theta', 'q', 'T', 'ele')
        save_dir: Directory to save plots
        filename: Base filename

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if filename is None:
        filename = f"multi_corridor_{data_key}"

    attr_map = {
        'alpha': 'alpha_grid', 'theta': 'theta_grid', 'q': 'q_grid',
        'T': 'T_grid', 'ele': 'ele_grid',
    }
    attr = attr_map.get(data_key, f"{data_key}_grid")

    colors_4 = [
        np.array([122, 187, 219]) / 255,
        np.array([132, 186, 66]) / 255,
        np.array([219, 180, 40]) / 255,
        np.array([212, 86, 46]) / 255,
    ]

    with _sci_style():
        fig = plt.figure(figsize=(16, 8))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')

        for k, (mesh, label) in enumerate(zip(mesh_list, labels)):
            data = getattr(mesh, attr).copy()
            data[mesh.flag_grid == 0] = np.nan

            color = colors_4[k % len(colors_4)]
            ax3d.plot_surface(
                mesh.V_grid, mesh.gamma_grid, data,
                color=color, alpha=0.7, edgecolor='none',
                label=label,
            )

        ax3d.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]", zlabel=data_key)
        ax3d.view_init(30, 45)

        # 2D heatmaps
        n = len(mesh_list)
        for k, (mesh, label) in enumerate(zip(mesh_list, labels)):
            ax = fig.add_subplot(2, n, n + k + 1)
            data = getattr(mesh, attr).copy()
            data[mesh.flag_grid == 0] = np.nan
            cmap = plt.cm.turbo.copy()
            cmap.set_bad(color='white')
            h = ax.pcolormesh(mesh.V_grid, mesh.gamma_grid, data,
                              cmap=cmap, shading='auto')
            ax.contour(mesh.V_grid, mesh.gamma_grid, mesh.flag_grid,
                       levels=[0.5], colors='k', linewidths=1, linestyles='--')
            fig.colorbar(h, ax=ax, shrink=0.8)
            ax.set_title(label, fontsize=9)
            ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]")

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved
