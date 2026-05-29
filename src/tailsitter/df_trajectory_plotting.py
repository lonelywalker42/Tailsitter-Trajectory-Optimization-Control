"""Visualization for differential-flatness-based trajectory results.

Replaces matlab/analysis/controller/visualize_trajectory.m,
visualize_simulation_results.m, visualize_transition_corridor.m,
visualize_multiple_trim_results.m, and visualize_transition_strategy.m.

Uses the _sci_style() context manager and COLORS palette from plotting.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .plotting import _sci_style, _box_on, COLORS
from .df_trim import DFTrimMesh
from .df_trajectory_optimization import DFTrajectoryResult
from .baseline_trajectory import BaselineTrajectoryResult
from .df_trajectory_simulation import SimulationResult, ErrorAnalysis
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
        ax.plot(x_pos[0], result.h[0], 'o', color='g', ms=8, mfc='g', zorder=5)
        ax.plot(x_pos[-1], result.h[-1], 'o', color=COLORS["red"], ms=8, mfc=COLORS["red"], zorder=5)
        ax.set(xlabel=r"$x$ [m]", ylabel=r"$h$ [m]", title="(a) Flight Trajectory")
        ax.legend(["Trajectory", "Start", "End"], fontsize=9)
        _box_on(ax)

        # (b) Velocity
        ax = axes[0, 1]
        ax.plot(t, result.V, color=COLORS["red"], lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$V$ [m/s]", title="(b) Velocity")
        _box_on(ax)

        # (c) Flight path angle (magenta, matching MATLAB 'm-')
        ax = axes[0, 2]
        ax.plot(t, gamma_deg, color="m", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\gamma$ [deg]", title="(c) Flight Path Angle")
        _box_on(ax)

        # (d) Alpha and theta
        ax = axes[1, 0]
        ax.plot(t, alpha_deg, color=COLORS["blue"], lw=2, label=r"$\alpha$")
        ax.plot(t, theta_deg, color=COLORS["red"], lw=2, ls="--", label=r"$\theta$")
        ax.set(xlabel="Time [s]", ylabel="Angle [deg]", title="(d) AoA and Pitch Angle")
        ax.legend(fontsize=9)
        _box_on(ax)

        # (e) Acceleration (green, matching MATLAB 'g-')
        ax = axes[1, 1]
        vdot = np.gradient(result.V, t)
        ax.plot(t, vdot, color="g", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\dot{V}$ [m/s$^2$]", title="(e) Acceleration")
        _box_on(ax)

        # (f) FPA rate (cyan, matching MATLAB 'c-')
        ax = axes[1, 2]
        gamma_rate = np.gradient(gamma_deg, t)
        ax.plot(t, gamma_rate, color="c", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\dot{\gamma}$ [deg/s]", title="(f) FPA Rate")
        _box_on(ax)

        # (g) Pitch rate (black, matching MATLAB 'k-')
        ax = axes[2, 0]
        ax.plot(t, q_deg, color="k", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$q$ [deg/s]", title="(g) Pitch Rate")
        _box_on(ax)

        # (h) Throttle
        ax = axes[2, 1]
        ax.plot(t, result.throttle, color=COLORS["orange"], lw=2)
        ax.set(xlabel="Time [s]", ylabel="Throttle [N]", title="(h) Throttle Input")
        _box_on(ax)

        # (i) Elevator
        ax = axes[2, 2]
        ax.plot(t, result.elevator, color="#7B1FA2", lw=2)
        ax.set(xlabel="Time [s]", ylabel=r"$\delta_e$ [deg]", title="(i) Elevator")
        _box_on(ax)

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
        ax.legend(fontsize=9)
        _box_on(ax)

        # (b) Flight path angle
        ax = axes[0, 1]
        ax.plot(t, np.degrees(gamma), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_gamma), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\gamma$ [deg]", title="(b) Flight Path Angle")
        _box_on(ax)

        # (c) Pitch angle
        ax = axes[0, 2]
        ax.plot(t, np.degrees(theta), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_theta), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\theta$ [deg]", title="(c) Pitch Angle")
        _box_on(ax)

        # (d) Angle of attack
        ax = axes[1, 0]
        ax.plot(t, np.degrees(alpha), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_alpha), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$\alpha$ [deg]", title="(d) Angle of Attack")
        _box_on(ax)

        # (e) Pitch rate
        ax = axes[1, 1]
        ax.plot(t, np.degrees(q), color=COLORS["blue"], lw=2)
        ax.plot(t, np.degrees(ref_q), color=COLORS["orange"], lw=1.5, ls="--")
        ax.set(xlabel="Time [s]", ylabel=r"$q$ [deg/s]", title="(e) Pitch Rate")
        _box_on(ax)

        # (f) V-gamma phase plane
        ax = axes[1, 2]
        ax.plot(V, np.degrees(gamma), color=COLORS["blue"], lw=2, label="Actual")
        ax.plot(ref_V, np.degrees(ref_gamma), color=COLORS["orange"], lw=1.5, ls="--", label="Reference")
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]", title=r"(f) V-$\gamma$ Phase")
        ax.legend(fontsize=9)
        _box_on(ax)

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

        # Use jet colormap to match MATLAB (parula for generic, jet for aero data)
        cmap = plt.cm.jet.copy()
        cmap.set_bad(color='white')

        h = ax.pcolormesh(mesh.V_grid, mesh.gamma_grid, data_plot,
                          cmap=cmap, shading='auto')
        if clim is not None:
            h.set_clim(*clim)

        # Contour lines (dark gray, matching MATLAB)
        with np.errstate(invalid='ignore'):
            cs = ax.contour(mesh.V_grid, mesh.gamma_grid, data_plot,
                            colors=[0.2, 0.2, 0.2], linewidths=1.2)
            ax.clabel(cs, fontsize=9)

        # Corridor boundary (red solid, matching MATLAB)
        ax.contour(mesh.V_grid, mesh.gamma_grid, mesh.flag_grid,
                   levels=[0.5], colors='r', linewidths=2)

        cb = fig.colorbar(h, ax=ax)
        cb.set_label(label, fontsize=12)
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]")
        _box_on(ax)

        # Data-dependent title
        _title_map = {
            r"$\alpha$ [deg]": r"(b) Angle of Attack $\alpha$ Distribution",
            r"$\theta$ [deg]": r"(c) Pitch Angle $\theta$ Distribution",
            r"$q$ [deg/s]": r"(a) Pitch Rate $q$ Distribution",
            r"$T$ [N]": r"(d) Thrust $T$ Distribution",
            r"$\delta_e$ [deg]": r"(e) Elevator Deflection $\delta_e$ Distribution",
            r"$q_{max}$ [deg/s]": r"(f) Max Pitch Rate $q_{max}$ Distribution",
        }
        title = _title_map.get(label, f"Transition Corridor — {label}")
        ax.set_title(title, fontsize=14)

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

        # Corridor boundary (red solid, matching MATLAB)
        ax.contour(mesh.V_grid, mesh.gamma_grid, mesh.flag_grid,
                   levels=[0.5], colors='r', linewidths=2,
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
        _box_on(ax)

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
                       levels=[0.5], colors='r', linewidths=1.5)
            fig.colorbar(h, ax=ax, shrink=0.8)
            ax.set_title(label, fontsize=11)
            ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]")

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Tracking error visualization (3×2 subplot)
# ---------------------------------------------------------------------------

def plot_tracking_error(
    sim: SimulationResult,
    error: ErrorAnalysis,
    save_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Plot trajectory tracking error analysis (3x2 subplot).

    Ported from matlab/analysis/controller/transition_sim_compare.m ::
    visualize_error_analysis(). Shows reference vs actual with error fill
    for V, gamma, theta, q, h, alpha.

    Args:
        sim: Simulation result
        error: Error analysis result
        save_dir: Directory to save plots
        prefix: Filename prefix (e.g., "proposed" or "baseline")

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    t = sim.time

    panels = [
        ("V", r"Velocity $V$ (m/s)", sim.ref_V, sim.actual_V,
         f"RMSE={error.rmse_V:.3f}", "(a) Velocity Tracking"),
        ("gamma", r"FPA $\gamma$ ($^\circ$)", sim.ref_gamma, sim.actual_gamma,
         f"RMSE={error.rmse_gamma:.3f}", "(b) Flight Path Angle Tracking"),
        ("theta", r"Pitch $\theta$ ($^\circ$)", sim.ref_theta, sim.actual_theta,
         f"RMSE={error.rmse_theta:.3f}", "(c) Pitch Angle Tracking"),
        ("q", r"Pitch Rate $q$ ($^\circ$/s)", sim.ref_q, sim.actual_q,
         f"RMSE={error.rmse_q:.3f}", "(d) Pitch Rate Tracking"),
        ("h", r"Altitude $h$ (m)", sim.ref_h, sim.actual_h,
         f"RMSE={error.rmse_height:.3f}", "(e) Altitude Tracking"),
        ("alpha", r"AoA $\alpha$ ($^\circ$)", sim.ref_alpha, sim.actual_alpha,
         f"RMSE={error.rmse_alpha:.3f}", "(f) Angle of Attack Tracking"),
    ]

    with _sci_style():
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

        for idx, (key, ylabel, ref, actual, rmse_str, title) in enumerate(panels):
            ax = axes[idx // 2, idx % 2]
            ax.plot(t, ref, "-", color=COLORS["blue"], linewidth=1.5, label="Reference")
            ax.plot(t, actual, "--", color=COLORS["red"], linewidth=1.5, label="Actual")
            ax.fill_between(t, ref, actual, alpha=0.2, color=COLORS["light_gray"],
                           label="Error")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best", frameon=False)
            ax.grid(True, alpha=0.3)
            _box_on(ax)
            ax.text(0.98, 0.95, rmse_str, transform=ax.transAxes,
                    fontsize=9, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        for ax in axes[2, :]:
            ax.set_xlabel(r"Time $t$ (s)")

        fig.tight_layout()
        fname = f"{prefix}error_analysis.png" if prefix else "error_analysis.png"
        path = save_dir / fname
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Transition strategy comparison (dual 2×3 windows)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryData:
    """Container for trajectory data used in transition strategy comparison."""
    time: np.ndarray
    V: np.ndarray          # m/s
    gamma: np.ndarray      # deg
    alpha: np.ndarray      # deg
    theta: np.ndarray      # deg
    q: np.ndarray          # deg/s
    T: np.ndarray          # N (thrust)
    delta: np.ndarray      # deg (elevator)
    q_margin: np.ndarray   # deg/s (pitch rate safety margin)


def _build_trajectory_data(
    traj_or_result,
    aero_data: dict | None = None,
) -> TrajectoryData:
    """Build TrajectoryData from DFTrajectoryResult or BaselineTrajectoryResult."""
    from .df_trajectory_optimization import DFTrajectoryResult as _DFTraj
    from .baseline_trajectory import BaselineTrajectoryResult as _BLTraj

    if isinstance(traj_or_result, _DFTraj):
        r = traj_or_result
        t = r.time
        gamma_deg = np.degrees(r.gamma)
        alpha_deg = np.degrees(r.alpha)
        theta_deg = np.degrees(r.theta)
        q_deg = np.degrees(r.q)
        q_margin = np.degrees(r.q_max) - np.abs(q_deg) if hasattr(r, 'q_max') and r.q_max is not None else np.full_like(q_deg, np.nan)
        return TrajectoryData(
            time=t, V=r.V, gamma=gamma_deg, alpha=alpha_deg,
            theta=theta_deg, q=q_deg, T=r.throttle,
            delta=r.elevator, q_margin=q_margin,
        )
    elif isinstance(traj_or_result, _BLTraj):
        r = traj_or_result
        path = r.optimal_path  # N×7: [V, gamma°, theta°, alpha°, T, ele°, qmax°]
        V = path[:, 0]
        gamma_deg = path[:, 1]
        theta_deg = path[:, 2]
        alpha_deg = path[:, 3]
        T = path[:, 4]
        delta = path[:, 5]
        qmax = path[:, 6]
        dV = np.diff(V)
        acc = 5.0
        dt_segments = np.abs(dV) / acc
        dt_segments = np.maximum(dt_segments, 0.001)
        t = np.concatenate([[0], np.cumsum(dt_segments)])
        q_deg = np.gradient(theta_deg, t)
        q_margin = qmax - np.abs(q_deg)
        return TrajectoryData(
            time=t, V=V, gamma=gamma_deg, alpha=alpha_deg,
            theta=theta_deg, q=q_deg, T=T, delta=delta,
            q_margin=q_margin,
        )
    else:
        raise TypeError(f"Unsupported type: {type(traj_or_result)}")


def plot_transition_strategy(
    proposed: TrajectoryData,
    baseline: TrajectoryData,
    save_dir: str | Path,
    filename_prefix: str = "transition_strategy",
) -> list[Path]:
    """Plot dual-window transition strategy comparison.

    Ported from matlab/analysis/controller/visualize_transition_strategy.m.

    Window 1 (2×3): Basic states — V, γ, V-γ phase, θ, α, q
    Window 2 (2×3): Rates & controls — V̇, γ̇, x-h trajectory, T, δe, q margin

    Args:
        proposed: Trajectory data for proposed method
        baseline: Trajectory data for baseline method
        save_dir: Directory to save plots
        filename_prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    c_proposed = (0.2, 0.4, 0.8)
    c_baseline = (0.8, 0.2, 0.2)
    c_safe = (0.9, 0.95, 1.0)
    c_zero = (0.5, 0.5, 0.5)
    c_start = (0.2, 0.7, 0.2)
    c_end = (0.7, 0.2, 0.2)

    p, b = proposed, baseline
    V_rate_p = np.gradient(p.V, p.time)
    V_rate_b = np.gradient(b.V, b.time)
    gamma_rate_p = np.gradient(p.gamma, p.time)
    gamma_rate_b = np.gradient(b.gamma, b.time)

    h_p = np.cumsum(p.V * np.sin(np.radians(p.gamma)) * np.gradient(p.time))
    x_p = np.cumsum(p.V * np.cos(np.radians(p.gamma)) * np.gradient(p.time))
    h_b = np.cumsum(b.V * np.sin(np.radians(b.gamma)) * np.gradient(b.time))
    x_b = np.cumsum(b.V * np.cos(np.radians(b.gamma)) * np.gradient(b.time))

    t_range = (min(p.time[0], b.time[0]), max(p.time[-1], b.time[-1]))

    # ====== Window 1: Basic States ======
    with _sci_style():
        fig1, axes1 = plt.subplots(2, 3, figsize=(14, 7))

        # (a) Velocity
        ax = axes1[0, 0]
        ax.plot(p.time, p.V, '-', color=c_proposed, lw=1.8, label="Proposed")
        ax.plot(b.time, b.V, '--', color=c_baseline, lw=1.8, label="Baseline")
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$V$ [m/s]", title="(a) Velocity")
        ax.set_xlim(t_range)
        ax.legend(fontsize=9, frameon=False)
        _box_on(ax)

        # (b) Flight path angle
        ax = axes1[0, 1]
        ax.plot(p.time, p.gamma, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.gamma, '--', color=c_baseline, lw=1.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\gamma$ [deg]", title="(b) Flight Path Angle")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (c) Phase plane V-γ
        ax = axes1[0, 2]
        ax.plot(p.V, p.gamma, '-', color=c_proposed, lw=2.0, label="Proposed")
        ax.plot(b.V, b.gamma, '--', color=c_baseline, lw=2.0, label="Baseline")
        ax.plot(p.V[0], p.gamma[0], 'o', color=c_start, ms=8, mfc=c_start, mec='k', zorder=5)
        ax.plot(p.V[-1], p.gamma[-1], 's', color=c_end, ms=8, mfc=c_end, mec='k', zorder=5)
        ax.plot(b.V[0], b.gamma[0], 'o', color=c_start, ms=8, mfc=c_start, mec='k', zorder=5)
        ax.plot(b.V[-1], b.gamma[-1], 's', color=c_end, ms=8, mfc=c_end, mec='k', zorder=5)
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]", title=r"(c) Phase Plane $(V, \gamma)$")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        _box_on(ax)

        # (d) Pitch angle
        ax = axes1[1, 0]
        ax.plot(p.time, p.theta, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.theta, '--', color=c_baseline, lw=1.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\theta$ [deg]", title="(d) Pitch Angle")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (e) Angle of attack (with zero line)
        ax = axes1[1, 1]
        ax.plot(p.time, p.alpha, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.alpha, '--', color=c_baseline, lw=1.8)
        ax.axhline(0, color=c_zero, ls=':', lw=0.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\alpha$ [deg]", title="(e) Angle of Attack")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (f) Pitch rate (with zero line)
        ax = axes1[1, 2]
        ax.plot(p.time, p.q, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.q, '--', color=c_baseline, lw=1.8)
        ax.axhline(0, color=c_zero, ls=':', lw=0.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$q$ [deg/s]", title="(f) Pitch Rate")
        ax.set_xlim(t_range)
        _box_on(ax)

        fig1.tight_layout()
        path1 = save_dir / f"{filename_prefix}_basic_states.png"
        fig1.savefig(path1, bbox_inches="tight", dpi=300)
        plt.close(fig1)
        saved.append(path1)

    # ====== Window 2: Rates & Controls ======
    with _sci_style():
        fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))

        # (a) Velocity rate
        ax = axes2[0, 0]
        ax.plot(p.time, V_rate_p, '-', color=c_proposed, lw=1.8, label="Proposed")
        ax.plot(b.time, V_rate_b, '--', color=c_baseline, lw=1.8, label="Baseline")
        ax.axhline(0, color=c_zero, ls=':', lw=0.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\dot{V}$ [m/s$^2$]", title="(a) Velocity Rate")
        ax.set_xlim(t_range)
        ax.legend(fontsize=9, frameon=False)
        _box_on(ax)

        # (b) FPA rate
        ax = axes2[0, 1]
        ax.plot(p.time, gamma_rate_p, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, gamma_rate_b, '--', color=c_baseline, lw=1.8)
        ax.axhline(0, color=c_zero, ls=':', lw=0.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\dot{\gamma}$ [deg/s]", title="(b) FPA Rate")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (c) x-h trajectory with flight direction arrows
        ax = axes2[0, 2]
        ax.plot(x_p, h_p, '-', color=c_proposed, lw=2.0, label="Proposed")
        ax.plot(x_b, h_b, '--', color=c_baseline, lw=2.0, label="Baseline")
        ax.plot(x_p[0], h_p[0], 'o', color=c_start, ms=8, mfc=c_start, mec='k', zorder=5)
        ax.plot(x_p[-1], h_p[-1], 's', color=c_end, ms=8, mfc=c_end, mec='k', zorder=5)
        ax.plot(x_b[0], h_b[0], 'o', color=c_start, ms=8, mfc=c_start, mec='k', zorder=5)
        ax.plot(x_b[-1], h_b[-1], 's', color=c_end, ms=8, mfc=c_end, mec='k', zorder=5)
        for xd, hd, c_arrow in [(x_p, h_p, c_proposed), (x_b, h_b, c_baseline)]:
            if len(xd) > 10:
                idx_a = len(xd) // 3
                ax.annotate('', xy=(xd[idx_a + 1], hd[idx_a + 1]),
                            xytext=(xd[idx_a], hd[idx_a]),
                            arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))
        ax.set(xlabel=r"$x$ [m]", ylabel=r"$h$ [m]", title="(c) Trajectory $(x, h)$")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        _box_on(ax)

        # (d) Thrust
        ax = axes2[1, 0]
        ax.plot(p.time, p.T, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.T, '--', color=c_baseline, lw=1.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$T$ [N]", title="(d) Thrust")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (e) Elevator
        ax = axes2[1, 1]
        ax.plot(p.time, p.delta, '-', color=c_proposed, lw=1.8)
        ax.plot(b.time, b.delta, '--', color=c_baseline, lw=1.8)
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"$\delta_e$ [deg]", title="(e) Elevator")
        ax.set_xlim(t_range)
        _box_on(ax)

        # (f) Pitch rate safety margin (area plot)
        ax = axes2[1, 2]
        ax.fill_between(p.time, 0, p.q_margin, color=c_safe,
                         edgecolor=c_proposed, linewidth=1.0, alpha=0.5, label="Proposed")
        ax.fill_between(b.time, 0, b.q_margin, color=c_safe,
                         edgecolor=c_baseline, linewidth=1.0, alpha=0.3, label="Baseline")
        ax.set(xlabel=r"Time $t$ [s]", ylabel=r"Margin [deg/s]",
               title="(f) Pitch Rate Safety Margin")
        ax.legend(fontsize=9, frameon=False)
        _box_on(ax)

        fig2.tight_layout()
        path2 = save_dir / f"{filename_prefix}_rates_controls.png"
        fig2.savefig(path2, bbox_inches="tight", dpi=300)
        plt.close(fig2)
        saved.append(path2)

    # ====== Performance comparison report ======
    report_lines = _compute_performance_report(
        p, b, V_rate_p, V_rate_b, gamma_rate_p, gamma_rate_b,
        h_p, h_b, x_p, x_b,
    )
    report_path = save_dir / f"{filename_prefix}_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    saved.append(report_path)

    return saved


def _compute_performance_report(
    p: TrajectoryData, b: TrajectoryData,
    V_rate_p, V_rate_b, gamma_rate_p, gamma_rate_b,
    h_p, h_b, x_p, x_b,
) -> list[str]:
    """Compute and format performance comparison report."""
    lines = []
    lines.append("=" * 60)
    lines.append("Transition Strategy Performance Comparison")
    lines.append("=" * 60)

    def _pct(proposed, baseline):
        if baseline == 0:
            return 0.0
        return 100.0 * (baseline - proposed) / baseline

    lines.append("")
    lines.append("--- Basic Parameters ---")
    lines.append(f"{'Parameter':<30} {'Proposed':>12} {'Baseline':>12} {'Change':>10}")
    lines.append("-" * 64)

    t_prop = p.time[-1] - p.time[0]
    t_base = b.time[-1] - b.time[0]
    lines.append(f"{'Total time [s]':<30} {t_prop:>12.2f} {t_base:>12.2f} {_pct(t_prop, t_base):>9.1f}%")

    dh_p = h_p[-1] - h_p[0]
    dh_b = h_b[-1] - h_b[0]
    lines.append(f"{'Height gain [m]':<30} {dh_p:>12.2f} {dh_b:>12.2f} {100*(dh_p-dh_b)/max(abs(dh_b),1e-6):>9.1f}%")

    dx_p = x_p[-1] - x_p[0]
    dx_b = x_b[-1] - x_b[0]
    lines.append(f"{'Horizontal dist [m]':<30} {dx_p:>12.2f} {dx_b:>12.2f} {_pct(dx_p, dx_b):>9.1f}%")

    lines.append("")
    lines.append("--- Flight State ---")
    lines.append(f"{'Parameter':<30} {'Proposed':>12} {'Baseline':>12} {'Change':>10}")
    lines.append("-" * 64)

    lines.append(f"{'Max alpha [deg]':<30} {np.max(p.alpha):>12.2f} {np.max(b.alpha):>12.2f} {_pct(np.max(p.alpha), np.max(b.alpha)):>9.1f}%")
    lines.append(f"{'Mean alpha [deg]':<30} {np.mean(p.alpha):>12.2f} {np.mean(b.alpha):>12.2f} {_pct(np.mean(p.alpha), np.mean(b.alpha)):>9.1f}%")
    lines.append(f"{'Max accel [m/s²]':<30} {np.max(np.abs(V_rate_p)):>12.2f} {np.max(np.abs(V_rate_b)):>12.2f} {_pct(np.max(np.abs(V_rate_p)), np.max(np.abs(V_rate_b))):>9.1f}%")
    lines.append(f"{'Max gamma rate [deg/s]':<30} {np.max(np.abs(gamma_rate_p)):>12.2f} {np.max(np.abs(gamma_rate_b)):>12.2f} {_pct(np.max(np.abs(gamma_rate_p)), np.max(np.abs(gamma_rate_b))):>9.1f}%")
    lines.append(f"{'Mean velocity [m/s]':<30} {np.mean(p.V):>12.2f} {np.mean(b.V):>12.2f} {100*(np.mean(p.V)-np.mean(b.V))/max(np.mean(b.V),1e-6):>9.1f}%")
    lines.append(f"{'V std [m/s]':<30} {np.std(p.V):>12.3f} {np.std(b.V):>12.3f} {_pct(np.std(p.V), np.std(b.V)):>9.1f}%")

    lines.append("")
    lines.append("--- Control Performance ---")
    lines.append(f"{'Parameter':<30} {'Proposed':>12} {'Baseline':>12} {'Change':>10}")
    lines.append("-" * 64)

    qm_mean_p = np.trapz(p.q_margin, p.time) / max(p.time[-1] - p.time[0], 1e-6)
    qm_mean_b = np.trapz(b.q_margin, b.time) / max(b.time[-1] - b.time[0], 1e-6)
    lines.append(f"{'Q margin mean [deg/s]':<30} {qm_mean_p:>12.2f} {qm_mean_b:>12.2f} {100*(qm_mean_p-qm_mean_b)/max(abs(qm_mean_b),1e-6):>9.1f}%")
    lines.append(f"{'Q margin min [deg/s]':<30} {np.nanmin(p.q_margin):>12.2f} {np.nanmin(b.q_margin):>12.2f} {100*(np.nanmin(p.q_margin)-np.nanmin(b.q_margin))/max(abs(np.nanmin(b.q_margin)),1e-6):>9.1f}%")
    lines.append(f"{'Max elevator [deg]':<30} {np.max(np.abs(p.delta)):>12.2f} {np.max(np.abs(b.delta)):>12.2f} {_pct(np.max(np.abs(p.delta)), np.max(np.abs(b.delta))):>9.1f}%")
    lines.append(f"{'Max thrust [N]':<30} {np.max(p.T):>12.1f} {np.max(b.T):>12.1f} {_pct(np.max(p.T), np.max(b.T)):>9.1f}%")

    # Composite score
    lines.append("")
    lines.append("--- Composite Score ---")
    w = {"safety": 0.3, "smooth": 0.25, "control": 0.2, "efficiency": 0.15, "time": 0.1}

    def _norm_score(v_p, v_b, higher_better=True):
        mn = min(v_p, v_b)
        mx = max(v_p, v_b)
        if mx - mn < 1e-9:
            return 0.5
        return (v_p - mn) / (mx - mn) if higher_better else (mx - v_p) / (mx - mn)

    s_safety = _norm_score(qm_mean_p, qm_mean_b, True)
    s_smooth = _norm_score(np.std(gamma_rate_p), np.std(gamma_rate_b), False)
    s_control = _norm_score(np.mean(np.abs(p.delta)), np.mean(np.abs(b.delta)), False)
    eff_p = dh_p / max(t_prop, 1e-6)
    eff_b = dh_b / max(t_base, 1e-6)
    s_eff = _norm_score(eff_p, eff_b, True)
    s_time = _norm_score(t_prop, t_base, False)

    score_p = (w["safety"] * s_safety + w["smooth"] * s_smooth +
               w["control"] * s_control + w["efficiency"] * s_eff +
               w["time"] * s_time)
    score_b = 1.0 - score_p

    lines.append(f"Proposed score:  {score_p:.3f}")
    lines.append(f"Baseline score:  {score_b:.3f}")
    if score_b > 1e-6:
        lines.append(f"Improvement:     {100*(score_p - score_b)/score_b:.1f}%")

    return lines


# ---------------------------------------------------------------------------
# Comprehensive simulation comparison (3×3)
# ---------------------------------------------------------------------------

def plot_simulation_comprehensive(
    sim_proposed: SimulationResult,
    save_dir: str | Path,
    sim_baseline: SimulationResult | None = None,
    filename: str = "sim_comprehensive",
) -> list[Path]:
    """Plot comprehensive 3×3 simulation comparison.

    Ported from matlab/analysis/controller/visualize_simulation_compare_results.m.

    Layout: V, γ, θ, α, q, h, throttle, elevator, V-γ phase.
    When sim_baseline is provided, overlays baseline traces.

    Args:
        sim_proposed: Simulation result for proposed trajectory
        sim_baseline: Optional simulation result for baseline trajectory
        save_dir: Directory to save plots
        filename: Base filename

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    sp = sim_proposed
    sb = sim_baseline

    with _sci_style():
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))

        # (a) Velocity
        ax = axes[0, 0]
        ax.plot(sp.time, sp.actual_V, color=COLORS["blue"], lw=2, label="Actual")
        ax.plot(sp.time, sp.ref_V, color=COLORS["orange"], lw=1.5, ls="--", label="Reference")
        if sb is not None:
            ax.plot(sb.time, sb.actual_V, color=COLORS["red"], lw=1.2, ls="-.", label="Baseline actual")
        ax.set(xlabel="Time [s]", ylabel=r"$V$ [m/s]", title="(a) Velocity")
        ax.legend(fontsize=8, frameon=False)
        _box_on(ax)

        # (b) Flight path angle
        ax = axes[0, 1]
        ax.plot(sp.time, sp.actual_gamma, color=COLORS["blue"], lw=2)
        ax.plot(sp.time, sp.ref_gamma, color=COLORS["orange"], lw=1.5, ls="--")
        if sb is not None:
            ax.plot(sb.time, sb.actual_gamma, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$\gamma$ [deg]", title="(b) Flight Path Angle")
        _box_on(ax)

        # (c) Pitch angle
        ax = axes[0, 2]
        ax.plot(sp.time, sp.actual_theta, color=COLORS["blue"], lw=2)
        ax.plot(sp.time, sp.ref_theta, color=COLORS["orange"], lw=1.5, ls="--")
        if sb is not None:
            ax.plot(sb.time, sb.actual_theta, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$\theta$ [deg]", title="(c) Pitch Angle")
        _box_on(ax)

        # (d) Angle of attack
        ax = axes[1, 0]
        ax.plot(sp.time, sp.actual_alpha, color=COLORS["blue"], lw=2)
        ax.plot(sp.time, sp.ref_alpha, color=COLORS["orange"], lw=1.5, ls="--")
        if sb is not None:
            ax.plot(sb.time, sb.actual_alpha, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$\alpha$ [deg]", title="(d) Angle of Attack")
        _box_on(ax)

        # (e) Pitch rate
        ax = axes[1, 1]
        ax.plot(sp.time, sp.actual_q, color=COLORS["blue"], lw=2)
        ax.plot(sp.time, sp.ref_q, color=COLORS["orange"], lw=1.5, ls="--")
        if sb is not None:
            ax.plot(sb.time, sb.actual_q, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$q$ [deg/s]", title="(e) Pitch Rate")
        _box_on(ax)

        # (f) Altitude
        ax = axes[1, 2]
        ax.plot(sp.time, sp.actual_h, color=COLORS["blue"], lw=2)
        ax.plot(sp.time, sp.ref_h, color=COLORS["orange"], lw=1.5, ls="--")
        if sb is not None:
            ax.plot(sb.time, sb.actual_h, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$h$ [m]", title="(f) Altitude")
        _box_on(ax)

        # (g) Throttle
        ax = axes[2, 0]
        ax.plot(sp.time, sp.throttle, color=COLORS["blue"], lw=2)
        if sb is not None:
            ax.plot(sb.time, sb.throttle, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel="Throttle", title="(g) Throttle")
        _box_on(ax)

        # (h) Elevator
        ax = axes[2, 1]
        ax.plot(sp.time, sp.elevator, color=COLORS["blue"], lw=2)
        if sb is not None:
            ax.plot(sb.time, sb.elevator, color=COLORS["red"], lw=1.2, ls="-.")
        ax.set(xlabel="Time [s]", ylabel=r"$\delta_e$ [deg]", title="(h) Elevator")
        _box_on(ax)

        # (i) V-γ phase plane
        ax = axes[2, 2]
        ax.plot(sp.actual_V, sp.actual_gamma, color=COLORS["blue"], lw=2, label="Actual")
        ax.plot(sp.ref_V, sp.ref_gamma, color=COLORS["orange"], lw=1.5, ls="--", label="Reference")
        if sb is not None:
            ax.plot(sb.actual_V, sb.actual_gamma, color=COLORS["red"], lw=1.2, ls="-.", label="Baseline")
        ax.set(xlabel=r"$V$ [m/s]", ylabel=r"$\gamma$ [deg]", title=r"(i) V-$\gamma$ Phase")
        ax.legend(fontsize=8, frameon=False)
        _box_on(ax)

        fig.tight_layout()
        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved
