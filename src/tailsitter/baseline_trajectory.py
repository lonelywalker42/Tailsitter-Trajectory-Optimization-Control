"""Baseline corridor path optimization for tailsitter UAV transitions.

Refactored from matlab/analysis/controller/trajectory_compare.m (baseline_main
and optimizeBoundaryPath functions). Finds an optimal transition path within
the feasibility corridor by maximizing distance from the corridor boundary.

This is the "baseline" method — a geometric path optimization in (V, gamma)
space, as opposed to the DF-based dynamic trajectory optimization in
df_trajectory_optimization.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from .df_trim import DFTrimMesh, df_trim_sweep
from .trim import R2D


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BaselineTrajectoryResult:
    """Result of baseline corridor path optimization."""
    # Optimized path: each row is [V, gamma_deg, theta_deg, alpha_deg, T, ele_deg, qmax_deg]
    optimal_path: np.ndarray = field(default_factory=lambda: np.array([]))
    # Corridor boundary points
    boundary_points: np.ndarray = field(default_factory=lambda: np.array([]))
    # Initial control points (before optimization)
    initial_control_points: np.ndarray = field(default_factory=lambda: np.array([]))
    # The trim mesh used
    df_trim_mesh: Optional[DFTrimMesh] = None
    # Start and end points
    start_point: tuple[float, float] = (0.1, 90.0)   # (V, gamma_deg)
    end_point: tuple[float, float] = (16.0, 0.0)


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _extract_boundary_points(
    V_grid: np.ndarray,
    gamma_grid: np.ndarray,
    feasibility_grid: np.ndarray,
) -> np.ndarray:
    """Extract boundary points from feasibility grid (contour at 0.5).

    Returns array of shape (n_points, 2) with [V, gamma] columns.
    """
    from matplotlib.contour import QuadContourSet
    import matplotlib.pyplot as plt

    # Use matplotlib's contour algorithm to extract the 0.5 isoline
    fig, ax = plt.subplots()
    cs = ax.contour(V_grid, gamma_grid, feasibility_grid, levels=[0.5])
    plt.close(fig)

    boundary_points = []
    for collection in cs.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            boundary_points.append(vertices)

    if boundary_points:
        return np.vstack(boundary_points)

    # Fallback: use grid edges
    V_min, V_max = V_grid.min(), V_grid.max()
    gamma_min, gamma_max = gamma_grid.min(), gamma_grid.max()
    n_edge = 100
    return np.vstack([
        np.column_stack([np.linspace(V_min, V_max, n_edge), np.full(n_edge, gamma_min)]),
        np.column_stack([np.full(n_edge, V_max), np.linspace(gamma_min, gamma_max, n_edge)]),
        np.column_stack([np.linspace(V_max, V_min, n_edge), np.full(n_edge, gamma_max)]),
        np.column_stack([np.full(n_edge, V_min), np.linspace(gamma_max, gamma_min, n_edge)]),
    ])


def _build_feasibility_interpolator(
    V_grid: np.ndarray,
    gamma_grid: np.ndarray,
    feasibility_grid: np.ndarray,
) -> RegularGridInterpolator:
    """Build a 2D interpolator for feasibility values."""
    V_vec = V_grid[0, :]
    gamma_vec = gamma_grid[:, 0]
    return RegularGridInterpolator(
        (gamma_vec, V_vec), feasibility_grid,
        method='linear', bounds_error=False, fill_value=0.0,
    )


def _adjust_to_feasible(
    point: np.ndarray,
    V_grid: np.ndarray,
    gamma_grid: np.ndarray,
    feasibility_grid: np.ndarray,
) -> np.ndarray:
    """Move a point to the nearest feasible location."""
    feasible_mask = feasibility_grid > 0.5
    if not np.any(feasible_mask):
        return point

    V_feasible = V_grid[feasible_mask]
    gamma_feasible = gamma_grid[feasible_mask]
    distances = np.sqrt((V_feasible - point[0])**2 + (gamma_feasible - point[1])**2)
    idx = np.argmin(distances)
    return np.array([V_feasible[idx], gamma_feasible[idx]])


def _initialize_control_points(
    start: np.ndarray,
    end: np.ndarray,
    n_points: int,
    V_grid: np.ndarray,
    gamma_grid: np.ndarray,
    feasibility_grid: np.ndarray,
    feas_interp: RegularGridInterpolator,
) -> np.ndarray:
    """Initialize control points along a straight line, adjusting to feasible region."""
    points = np.zeros((n_points, 2))
    points[0] = start
    points[-1] = end

    for i in range(1, n_points - 1):
        t = i / (n_points - 1)
        candidate = (1 - t) * start + t * end
        feas = feas_interp([[candidate[1], candidate[0]]])[0]
        if feas <= 0.5:
            candidate = _adjust_to_feasible(candidate, V_grid, gamma_grid, feasibility_grid)
        points[i] = candidate

    return points


# ---------------------------------------------------------------------------
# Path optimization
# ---------------------------------------------------------------------------

def _optimize_control_points(
    initial_points: np.ndarray,
    boundary_points: np.ndarray,
    V_grid: np.ndarray,
    gamma_grid: np.ndarray,
    feasibility_grid: np.ndarray,
    feas_interp: RegularGridInterpolator,
) -> tuple[np.ndarray, float]:
    """Optimize internal control points to maximize distance from boundary."""
    n_points = len(initial_points)
    internal_idx = slice(1, n_points - 1)
    x0 = initial_points[internal_idx].flatten()

    V_range = [V_grid.min(), V_grid.max()]
    gamma_range = [gamma_grid.min(), gamma_grid.max()]
    n_internal = n_points - 2
    lb = np.tile([V_range[0], gamma_range[0]], n_internal)
    ub = np.tile([V_range[1], gamma_range[1]], n_internal)

    def objective(x):
        points = initial_points.copy()
        points[internal_idx] = x.reshape(-1, 2)
        distances = np.array([
            np.min(np.sqrt(np.sum((boundary_points - p)**2, axis=1)))
            for p in points
        ])
        # Maximize distance = minimize negative distance
        f = -np.sum(distances)

        # Smoothness penalty
        lam = 0.1
        smoothness = 0.0
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i - 1]
            v2 = points[i + 1] - points[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                smoothness += np.arccos(cos_a)**2
        return f + lam * smoothness

    def constraint_ineq(x):
        points = initial_points.copy()
        points[internal_idx] = x.reshape(-1, 2)
        c = []
        # Segment length constraints
        for i in range(len(points) - 1):
            seg_len = np.linalg.norm(points[i + 1] - points[i])
            c.append(seg_len - 0.1)   # max segment length
            c.append(0.01 - seg_len)  # min segment length
        # Feasibility constraints
        for p in points:
            feas = feas_interp([[p[1], p[0]]])[0]
            c.append(0.01 - feas)
        return np.array(c)

    from scipy.optimize import NonlinearConstraint
    nl_con = NonlinearConstraint(constraint_ineq, -np.inf, 0)

    result = minimize(
        objective, x0, method='trust-constr',
        bounds=list(zip(lb, ub)),
        constraints=[nl_con],
        options={'maxiter': 2000, 'verbose': 0},
    )

    optimal_points = initial_points.copy()
    optimal_points[internal_idx] = result.x.reshape(-1, 2)
    return optimal_points, result.fun


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def optimize_baseline_path(
    df_trim_mesh: DFTrimMesh,
    start_point: tuple[float, float] = (0.1, 90.0),
    end_point: tuple[float, float] = (16.0, 0.0),
    n_control_points: int = 16,
    verbose: bool = True,
) -> BaselineTrajectoryResult:
    """Optimize a baseline transition path within the feasibility corridor.

    Ported from matlab/analysis/controller/trajectory_compare.m (baseline_main).

    Args:
        df_trim_mesh: DF corridor sweep result
        start_point: (V_start, gamma_start_deg)
        end_point: (V_end, gamma_end_deg)
        n_control_points: Number of control points including start/end
        verbose: Print progress

    Returns:
        BaselineTrajectoryResult with optimized path and corridor data
    """
    V_grid = df_trim_mesh.V_grid
    gamma_grid = df_trim_mesh.gamma_grid

    # Build feasibility grid (same limits as MATLAB SinglePlot)
    feasibility_grid = (
        df_trim_mesh.flag_grid
        * (df_trim_mesh.T_grid >= 0) * (df_trim_mesh.T_grid <= 140)
        * (df_trim_mesh.ele_grid >= -20) * (df_trim_mesh.ele_grid <= 30)
        * (df_trim_mesh.alpha_grid >= -10) * (df_trim_mesh.alpha_grid <= 16)
        * (df_trim_mesh.V_grid <= 16) * (df_trim_mesh.V_grid >= 0.1)
    ).astype(float)

    feas_interp = _build_feasibility_interpolator(V_grid, gamma_grid, feasibility_grid)

    # Extract boundary
    boundary_points = _extract_boundary_points(V_grid, gamma_grid, feasibility_grid)
    if verbose:
        print(f"  Boundary points: {len(boundary_points)}")

    # Initialize control points
    start = np.array(start_point)
    end = np.array(end_point)
    initial_points = _initialize_control_points(
        start, end, n_control_points, V_grid, gamma_grid, feasibility_grid, feas_interp,
    )

    if verbose:
        print(f"  Initial control points: {n_control_points}")

    # Optimize
    optimal_points, fval = _optimize_control_points(
        initial_points, boundary_points, V_grid, gamma_grid, feasibility_grid, feas_interp,
    )

    if verbose:
        print(f"  Optimization complete, objective: {fval:.4f}")

    # Interpolate trim data along optimal path
    V_path = optimal_points[:, 0]
    gamma_path = optimal_points[:, 1]

    theta_interp = RegularGridInterpolator(
        (df_trim_mesh.trim_gamma, df_trim_mesh.trim_V), df_trim_mesh.theta_grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    alpha_interp = RegularGridInterpolator(
        (df_trim_mesh.trim_gamma, df_trim_mesh.trim_V), df_trim_mesh.alpha_grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    T_interp = RegularGridInterpolator(
        (df_trim_mesh.trim_gamma, df_trim_mesh.trim_V), df_trim_mesh.T_grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    ele_interp = RegularGridInterpolator(
        (df_trim_mesh.trim_gamma, df_trim_mesh.trim_V), df_trim_mesh.ele_grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    qmax_interp = RegularGridInterpolator(
        (df_trim_mesh.trim_gamma, df_trim_mesh.trim_V), df_trim_mesh.qmax_grid,
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    # Query interpolators
    query_points = np.column_stack([gamma_path, V_path])
    theta_path = theta_interp(query_points)
    alpha_path = alpha_interp(query_points)
    T_path = T_interp(query_points)
    ele_path = ele_interp(query_points)
    qmax_path = qmax_interp(query_points)

    # Build optimal_path: [V, gamma_deg, theta_deg, alpha_deg, T, ele_deg, qmax_deg]
    optimal_path = np.column_stack([
        V_path, gamma_path, theta_path, alpha_path, T_path, ele_path, qmax_path,
    ])

    if verbose:
        print(f"  Path points: {len(optimal_path)}")
        for i, row in enumerate(optimal_path):
            print(f"    {i}: V={row[0]:.2f}, gamma={row[1]:.2f}, "
                  f"theta={row[2]:.2f}, alpha={row[3]:.2f}")

    return BaselineTrajectoryResult(
        optimal_path=optimal_path,
        boundary_points=boundary_points,
        initial_control_points=initial_points,
        df_trim_mesh=df_trim_mesh,
        start_point=start_point,
        end_point=end_point,
    )


def save_baseline_result(result: BaselineTrajectoryResult, path: str | Path) -> None:
    """Save BaselineTrajectoryResult to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        optimal_path=result.optimal_path,
        boundary_points=result.boundary_points,
        initial_control_points=result.initial_control_points,
        start_point=np.array(result.start_point),
        end_point=np.array(result.end_point),
    )


def load_baseline_result(path: str | Path, df_trim_mesh: Optional[DFTrimMesh] = None) -> BaselineTrajectoryResult:
    """Load BaselineTrajectoryResult from .npz file."""
    data = np.load(str(path))
    return BaselineTrajectoryResult(
        optimal_path=data["optimal_path"],
        boundary_points=data["boundary_points"],
        initial_control_points=data["initial_control_points"],
        start_point=tuple(data["start_point"]),
        end_point=tuple(data["end_point"]),
        df_trim_mesh=df_trim_mesh,
    )
