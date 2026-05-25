"""Visualization for trim analysis results.

SCI journal style plots matching matlab/trim/CorridorPlot.m,
StaticCorridorPlot.m, and linearization/CorridorPlot.m.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .plotting import _sci_style, COLORS
from .trim import TrimMesh


# ---------------------------------------------------------------------------
# Corridor plots
# ---------------------------------------------------------------------------

def plot_corridor(trimmesh: TrimMesh, save_dir: str | Path) -> list[str]:
    """Generate transition corridor plots.

    Maps CorridorPlot.m — produces 7 figures:
    1. Transition corridor boundary
    2. Trim alpha heatmap
    3. Trim gamma heatmap
    4. Trim elevator heatmap
    5. Trim throttle heatmap
    6. Forward transition residual moment heatmap
    7. Backward transition residual moment heatmap

    Args:
        trimmesh: TrimMesh with sweep results
        save_dir: Directory to save figures

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, Y = np.meshgrid(trimmesh.trim_V, trimmesh.trim_theta)
    trim_grid = trimmesh.trim_flag.T  # transpose for plotting (theta on y-axis)

    saved = []

    # Prepare masked arrays (NaN where not trimmable)
    alpha_grid = trimmesh.alpha.T.copy()
    alpha_grid[trim_grid == 0] = np.nan

    gamma_grid = trimmesh.gamma.T.copy()
    gamma_grid[trim_grid == 0] = np.nan

    ele_grid = trimmesh.elevator.T.copy()
    ele_grid[trim_grid == 0] = np.nan

    throttle_grid = trimmesh.throttle.T.copy()
    throttle_grid[trim_grid == 0] = np.nan

    ft_grid = trimmesh.max_moment_ft.T.copy()
    ft_grid[trim_grid == 0] = np.nan

    bt_grid = trimmesh.max_moment_bt.T.copy()
    bt_grid[trim_grid == 0] = np.nan

    # Figure 1: Transition corridor boundary
    path = _plot_corridor_boundary(X, Y, trim_grid, save_dir, "corridor_boundary")
    saved.append(path)

    # Figure 2: Alpha heatmap
    path = _plot_heatmap(trimmesh, alpha_grid, trim_grid, r"$\alpha$ [deg]",
                         save_dir, "corridor_alpha", title="Trim Angle of Attack")
    saved.append(path)

    # Figure 3: Gamma heatmap
    path = _plot_heatmap(trimmesh, gamma_grid, trim_grid, r"$\gamma$ [deg]",
                         save_dir, "corridor_gamma", title="Flight Path Angle")
    saved.append(path)

    # Figure 4: Elevator heatmap
    path = _plot_heatmap(trimmesh, ele_grid, trim_grid, r"$\delta_e$ [deg]",
                         save_dir, "corridor_elevator", title="Elevator Deflection")
    saved.append(path)

    # Figure 5: Throttle heatmap
    path = _plot_heatmap(trimmesh, throttle_grid, trim_grid, r"$\delta_t$",
                         save_dir, "corridor_throttle", title="Throttle")
    saved.append(path)

    # Figure 6: Forward transition moment
    path = _plot_heatmap(trimmesh, ft_grid, trim_grid, "Residual Moment [N·m]",
                         save_dir, "corridor_moment_ft",
                         title="Residual Moment (Forward Transition)")
    saved.append(path)

    # Figure 7: Backward transition moment
    path = _plot_heatmap(trimmesh, bt_grid, trim_grid, "Residual Moment [N·m]",
                         save_dir, "corridor_moment_bt",
                         title="Residual Moment (Backward Transition)")
    saved.append(path)

    return saved


def _plot_corridor_boundary(X, Y, trim_grid, save_dir, filename) -> str:
    """Plot corridor boundary with fill."""
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Contour at 0.5 to get boundary
        cs = ax.contour(X, Y, trim_grid, levels=[0.5], linewidths=1.5,
                        colors=[COLORS["gray"]])

        # Fill the trimmable area
        if len(cs.allsegs[0]) > 0:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection
            for seg in cs.allsegs[0]:
                poly = Polygon(seg, closed=True,
                               facecolor=COLORS["light_gray"],
                               alpha=0.3, edgecolor='none')
                ax.add_patch(poly)

        ax.set_xlabel(r"$V$ [m/s]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_title("Transition Corridor")
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 90])
        ax.set_axisbelow(True)

        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return str(path)


def _plot_heatmap(trimmesh: TrimMesh, data_grid: np.ndarray,
                  trim_grid: np.ndarray, cbar_label: str,
                  save_dir: Path, filename: str,
                  title: str = "") -> str:
    """Plot a single heatmap with corridor boundary overlay."""
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='white')

        masked = np.ma.masked_invalid(data_grid)
        im = ax.pcolormesh(trimmesh.trim_V, trimmesh.trim_theta, masked,
                           cmap=cmap, shading='auto')

        # Overlay corridor boundary
        X, Y = np.meshgrid(trimmesh.trim_V, trimmesh.trim_theta)
        ax.contour(X, Y, trim_grid, levels=[0.5], linewidths=1.0,
                   colors='k', linestyles='--')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=10)

        ax.set_xlabel(r"$V$ [m/s]")
        ax.set_ylabel(r"$\theta$ [deg]")
        if title:
            ax.set_title(title)
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 90])

        path = save_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return str(path)


# ---------------------------------------------------------------------------
# Level flight plots
# ---------------------------------------------------------------------------

def plot_level_flight(trim_result: dict, save_dir: str | Path) -> list[str]:
    """Plot level flight trim results vs velocity.

    Maps trimLF.m figure generation.

    Args:
        trim_result: Dict from sweep_level_flight with keys:
                     V, flag, throttle, elevator, alpha, theta
        save_dir: Directory to save figures

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    V = trim_result['V']
    flag = trim_result['flag']
    mask = flag == 1

    saved = []

    # Theta vs V
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(V[mask], trim_result['theta'][mask], color=COLORS["blue"],
                linewidth=1.2)
        ax.set_xlabel(r"$V$ [m/s]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_title("Level Flight Trim: Pitch Angle")
        ax.set_axisbelow(True)
        path = save_dir / "level_flight_theta.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(str(path))

    # Elevator vs V
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(V[mask], trim_result['elevator'][mask], color=COLORS["blue"],
                linewidth=1.2)
        ax.set_xlabel(r"$V$ [m/s]")
        ax.set_ylabel(r"$\delta_e$ [deg]")
        ax.set_title("Level Flight Trim: Elevator")
        ax.set_axisbelow(True)
        path = save_dir / "level_flight_elevator.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(str(path))

    # Throttle vs V
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(V[mask], trim_result['throttle'][mask], color=COLORS["blue"],
                linewidth=1.2)
        ax.set_xlabel(r"$V$ [m/s]")
        ax.set_ylabel(r"$\delta_t$")
        ax.set_title("Level Flight Trim: Throttle")
        ax.set_axisbelow(True)
        path = save_dir / "level_flight_throttle.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        saved.append(str(path))

    return saved


# ---------------------------------------------------------------------------
# Eigenvalue corridor plots
# ---------------------------------------------------------------------------

def plot_eigenvalue_corridor(trimmesh: TrimMesh,
                             save_dir: str | Path) -> list[str]:
    """Plot eigenvalue/controllability heatmaps over the corridor.

    Maps linearization/CorridorPlot.m — produces 5 figures:
    1. Transition corridor boundary
    2. Real eigenvalue count
    3. Max real part
    4. Min real part
    5. Gramian min singular value

    Args:
        trimmesh: TrimMesh with linearization data
        save_dir: Directory to save figures

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trim_grid = trimmesh.trim_flag.T
    saved = []

    # Figure 1: Corridor boundary
    X, Y = np.meshgrid(trimmesh.trim_V, trimmesh.trim_theta)
    path = _plot_corridor_boundary(X, Y, trim_grid, save_dir,
                                   "eigen_corridor_boundary")
    saved.append(path)

    if trimmesh.real_max is None:
        return saved

    # Figure 2: Real eigenvalue count
    data = trimmesh.real_num.T.copy().astype(float)
    data[trim_grid == 0] = np.nan
    path = _plot_heatmap(trimmesh, data, trim_grid, "Real Eigenvalue Count",
                         save_dir, "eigen_real_count",
                         title="Real Eigenvalue Count")
    saved.append(path)

    # Figure 3: Max real part
    data = trimmesh.real_max.T.copy()
    data[trim_grid == 0] = np.nan
    path = _plot_heatmap(trimmesh, data, trim_grid, "Max Real Part",
                         save_dir, "eigen_real_max",
                         title="Maximum Eigenvalue Real Part")
    saved.append(path)

    # Figure 4: Min real part
    data = trimmesh.real_min.T.copy()
    data[trim_grid == 0] = np.nan
    path = _plot_heatmap(trimmesh, data, trim_grid, "Min Real Part",
                         save_dir, "eigen_real_min",
                         title="Minimum Eigenvalue Real Part")
    saved.append(path)

    # Figure 5: Gramian min singular value
    if trimmesh.min_singular_value is not None:
        data = trimmesh.min_singular_value.T.copy()
        data[trim_grid == 0] = np.nan
        path = _plot_heatmap(trimmesh, data, trim_grid,
                             "Gramian Min Singular Value",
                             save_dir, "eigen_msv",
                             title="Controllability Gramian Min SV")
        saved.append(path)

    return saved
