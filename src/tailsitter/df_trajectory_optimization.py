"""Differential-flatness-based trajectory optimization.

Refactored from matlab/analysis/controller/trajectory_plan.m — optimizes
time increments (dt) and flight path angle increments (dgamma) along a
velocity grid to minimize a weighted cost of time, height change, control
energy, and pitch rate safety margin.

Uses differential_flatness_transfer internally to convert flat outputs
(V, gamma) to full state/control at each grid point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from .differential_flatness import (
    AeroSin4Coeffs,
    DFTransferResult,
    PitchRateMargin,
    PitchRateMarginParams,
    calculate_max_feasible_pitch_rate,
    differential_flatness_transfer,
    load_sin4_coeffs,
)
from .df_trim import DFTrimMesh, df_trim_sweep
from .trim import (
    R2D, G, RHO, S_REF, C_REF, M_TOTAL, MS, PROP_R,
    AeroData, load_aero_data,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DFTrajectoryResult:
    """Result of DF-based trajectory optimization."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    V: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))   # rad
    h: np.ndarray = field(default_factory=lambda: np.array([]))       # m
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))   # rad
    theta: np.ndarray = field(default_factory=lambda: np.array([]))   # rad
    q: np.ndarray = field(default_factory=lambda: np.array([]))       # rad/s
    throttle: np.ndarray = field(default_factory=lambda: np.array([]))
    elevator: np.ndarray = field(default_factory=lambda: np.array([]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([]))   # rad/s
    cost: float = 0.0
    solve_time: float = 0.0
    success: bool = False
    exitflag: int = 0


@dataclass
class TrajectoryOptParams:
    """Parameters for trajectory optimization."""
    w_time: float = 0.1         # time weight
    w_height: float = 0.0       # height change weight
    w_energy: float = 0.1       # control energy weight
    w_safety: float = 0.1       # pitch rate safety margin weight
    max_gamma_rate: float = 30.0    # max FPA rate [deg/s]
    max_alpha: float = 10.0         # max AoA [deg]
    max_theta_rate: float = 30.0    # max pitch rate [deg/s]
    dv: float = 1.0                 # velocity grid step [m/s]
    min_dt: float = 0.1             # min time per segment [s]
    max_dt: float = 20.0            # max time per segment [s]
    smoothness_lambda: float = 0.1  # smoothness regularization coefficient
    limit: tuple[float, float] = (-20.0, 30.0)  # elevator limits
    throttle0: float = 63.7        # initial throttle guess


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def _compute_trajectory(
    dt: np.ndarray,
    dgamma: np.ndarray,
    v_grid: np.ndarray,
    gamma0: float,
    h0: float,
    sin4_coeffs: AeroSin4Coeffs,
    aero: AeroData,
    phys_kwargs: dict,
    margin_params: Optional[PitchRateMarginParams],
    limit: tuple[float, float],
    throttle0: float,
) -> tuple[np.ndarray, ...]:
    """Compute full trajectory from dt and dgamma increments.

    Returns: (time, gamma, h, alpha, theta, q, v, throttle, elevator, q_max)
    """
    N = len(dt)
    time = np.zeros(N + 1)
    gamma = np.zeros(N + 1)
    h = np.zeros(N + 1)
    alpha = np.zeros(N + 1)
    theta = np.zeros(N + 1)
    q = np.zeros(N + 1)
    v = np.zeros(N + 1)
    throttle = np.zeros(N + 1)
    elevator = np.zeros(N + 1)
    q_max = np.zeros(N + 1)

    # Initial conditions
    time[0] = 0.0
    gamma[0] = gamma0
    h[0] = h0
    v[0] = v_grid[0]
    theta[0] = gamma0 + alpha[0]
    throttle[0] = throttle0
    elevator[0] = 0.0

    for i in range(N):
        time[i + 1] = time[i] + dt[i]
        v[i + 1] = v_grid[i + 1]
        gamma[i + 1] = gamma[i] + dgamma[i]
        h[i + 1] = h[i] + v[i] * np.sin(gamma[i]) * dt[i]

        # Differential flatness transfer
        Vdot = (v[i + 1] - v[i]) / dt[i]
        gammadot_seg = dgamma[i] / dt[i]

        result = differential_flatness_transfer(
            V=v[i + 1], Vdot=Vdot, Vddot=0.0, Vdddot=0.0,
            gamma=gamma[i + 1], gammadot=gammadot_seg, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero, **phys_kwargs,
        )

        alpha[i + 1] = result.alpha
        theta[i + 1] = result.theta
        q[i + 1] = result.q
        throttle[i + 1] = result.T
        elevator[i + 1] = result.elevator

        margin = calculate_max_feasible_pitch_rate(
            V=v[i + 1], gamma=gamma[i + 1],
            V_dot=Vdot, gamma_dot=gammadot_seg,
            delta_e_flat=result.elevator, T_flat=result.T,
            alpha_flat=result.alpha, limit=limit,
            params=margin_params,
        )
        q_max[i + 1] = margin.q_max

    return time, gamma, h, alpha, theta, q, v, throttle, elevator, q_max


# ---------------------------------------------------------------------------
# Objective and constraints
# ---------------------------------------------------------------------------

def _objective(
    x: np.ndarray, N: int,
    v_grid: np.ndarray, gamma0: float, h0: float,
    sin4_coeffs: AeroSin4Coeffs, aero: AeroData,
    phys_kwargs: dict, margin_params: Optional[PitchRateMarginParams],
    params: TrajectoryOptParams,
) -> float:
    """Weighted cost: time + height + energy + safety + smoothness."""
    dt = x[:N]
    dgamma = x[N:]

    time, gamma, h, _, _, q, v, throttle, _, q_max = _compute_trajectory(
        dt, dgamma, v_grid, gamma0, h0,
        sin4_coeffs, aero, phys_kwargs, margin_params,
        params.limit, params.throttle0,
    )

    total_time = np.sum(dt)
    height_change = np.sum(h**2)
    energy = np.sum(throttle**2)
    safety = np.sum(np.degrees(q - q_max)**2)

    # Normalization factors
    time_norm = 1.0
    height_norm = 100.0
    energy_norm = 20000.0
    safety_norm = 900.0

    J = (params.w_time * (total_time / time_norm)
         + params.w_height * (height_change / height_norm)
         + params.w_energy * (energy / energy_norm)
         - params.w_safety * (safety / safety_norm))

    # Smoothness regularization (angle between consecutive path segments)
    points = np.column_stack([v / 16.0, gamma / 90.0])
    lam = params.smoothness_lambda * R2D**2
    smoothness = 0.0
    for i in range(1, len(points) - 1):
        vec1 = points[i] - points[i - 1]
        vec2 = points[i + 1] - points[i]
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.clip(np.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(cos_a)
            smoothness += angle**2

    return J + lam * smoothness


def _constraints(
    x: np.ndarray, N: int,
    v_grid: np.ndarray, gamma0: float, gammaf: float,
    sin4_coeffs: AeroSin4Coeffs, aero: AeroData,
    phys_kwargs: dict, margin_params: Optional[PitchRateMarginParams],
    params: TrajectoryOptParams,
) -> np.ndarray:
    """Inequality constraints (all <= 0)."""
    dt = x[:N]
    dgamma = x[N:]

    time, gamma, _, alpha, _, q, _, throttle, _, q_max = _compute_trajectory(
        dt, dgamma, v_grid, gamma0, 0.0,
        sin4_coeffs, aero, phys_kwargs, margin_params,
        params.limit, params.throttle0,
    )

    max_gr = np.radians(params.max_gamma_rate)
    max_a = np.radians(params.max_alpha)
    max_qr = np.radians(params.max_theta_rate)

    c = []

    # Gamma rate constraints
    gamma_rate = dgamma / dt
    c.extend(np.abs(gamma_rate) - max_gr)

    # Gamma bounds
    c.extend(-gamma)            # gamma >= 0
    c.extend(gamma - np.pi / 2)  # gamma <= pi/2

    # Terminal gamma
    c.append(abs(gamma[-1] - gammaf) - 0.1 / R2D)

    # Alpha constraints
    c.extend(np.abs(alpha) - max_a)

    # Pitch rate constraints
    c.extend(np.abs(q) - max_qr)
    c.extend(np.abs(q) - np.abs(q_max))

    # Throttle non-negative
    c.extend(-throttle)

    return np.array(c)


def _eq_constraints(
    x: np.ndarray, N: int,
    v_grid: np.ndarray, gamma0: float,
) -> np.ndarray:
    """Equality constraints: gamma continuity."""
    dgamma = x[N:]
    gamma = np.zeros(N + 1)
    gamma[0] = gamma0
    for i in range(N):
        gamma[i + 1] = gamma[i] + dgamma[i]

    ceq = []
    for i in range(1, N + 1):
        ceq.append(gamma[i] - gamma[i - 1] - dgamma[i - 1])
    return np.array(ceq)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def df_trajectory_optimize(
    vel: tuple[float, float],
    gamma_deg: tuple[float, float],
    params: Optional[TrajectoryOptParams] = None,
    config_name: str = "aero_cfg2",
    data_dir: str | Path = "data/processed",
    margin_params: Optional[PitchRateMarginParams] = None,
    verbose: bool = True,
) -> DFTrajectoryResult:
    """Run DF-based trajectory optimization.

    Ported from matlab/analysis/controller/trajectory_plan.m.

    Args:
        vel: (V_start, V_end) in m/s
        gamma_deg: (gamma_start, gamma_end) in degrees
        params: Optimization parameters
        config_name: Aero config name
        data_dir: Path to processed data
        margin_params: Pitch rate margin tuning
        verbose: Print progress

    Returns:
        DFTrajectoryResult with optimized trajectory
    """
    import time as _time

    if params is None:
        params = TrajectoryOptParams()

    # Load data
    aero = load_aero_data(config_name, data_dir)
    sin4_coeffs = load_sin4_coeffs(config_name, data_dir)
    phys_kwargs = dict(m=M_TOTAL, g=G, rho=RHO, S=S_REF, c_bar=C_REF, Jy=0.183, prop_R=PROP_R)

    v0, vf = vel
    gamma0 = np.radians(gamma_deg[0])
    gammaf = np.radians(gamma_deg[1])
    h0 = 0.0

    # Velocity grid
    if vf < v0:
        v_grid = np.arange(vf, v0 + params.dv * 0.5, params.dv)
        v_grid = v_grid[::-1]
    else:
        v_grid = np.arange(v0, vf + params.dv * 0.5, params.dv)

    N = len(v_grid) - 1
    n_vars = 2 * N

    if verbose:
        print(f"DF trajectory optimization: {N} segments, {n_vars} variables")
        print(f"  V: {v0:.1f} -> {vf:.1f} m/s, gamma: {gamma_deg[0]:.1f} -> {gamma_deg[1]:.1f} deg")

    # Initial guess
    x0 = np.zeros(n_vars)
    x0[:N] = 0.5  # dt
    x0[N:] = (gammaf - gamma0) / N  # dgamma

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    lb[:N] = params.min_dt
    ub[:N] = params.max_dt
    max_gr = np.radians(params.max_gamma_rate)
    lb[N:] = -max_gr * params.max_dt
    ub[N:] = max_gr * params.max_dt
    bounds = list(zip(lb, ub))

    # Wrappers for optimizer
    def obj(x):
        return _objective(x, N, v_grid, gamma0, h0, sin4_coeffs, aero, phys_kwargs, margin_params, params)

    def ineq(x):
        return _constraints(x, N, v_grid, gamma0, gammaf, sin4_coeffs, aero, phys_kwargs, margin_params, params)

    def eq(x):
        return _eq_constraints(x, N, v_grid, gamma0)

    nonlinear_ineq = NonlinearConstraint(ineq, -np.inf, 0)
    nonlinear_eq = NonlinearConstraint(eq, 0, 0)

    # Solve
    if verbose:
        print("  Starting optimization...")

    t0 = _time.time()
    result = minimize(
        obj, x0, method='trust-constr',
        bounds=bounds,
        constraints=[nonlinear_ineq, nonlinear_eq],
        options={'maxiter': 1000, 'verbose': 2 if verbose else 0},
    )
    solve_time = _time.time() - t0

    # Extract results
    x_opt = result.x
    dt_opt = x_opt[:N]
    dgamma_opt = x_opt[N:]

    time, gamma, h, alpha, theta, q, v, throttle, elevator, q_max = _compute_trajectory(
        dt_opt, dgamma_opt, v_grid, gamma0, h0,
        sin4_coeffs, aero, phys_kwargs, margin_params,
        params.limit, params.throttle0,
    )

    if verbose:
        print(f"\n  Optimization {'succeeded' if result.success else 'FAILED'}")
        print(f"  Solve time: {solve_time:.2f}s, Cost: {result.fun:.6f}")
        print(f"  Total time: {time[-1]:.2f}s, Height change: {h[-1] - h[0]:.2f}m")
        print(f"  Max alpha: {np.degrees(np.max(np.abs(alpha))):.2f} deg")
        print(f"  Max |q|: {np.degrees(np.max(np.abs(q))):.2f} deg/s")

    return DFTrajectoryResult(
        time=time, V=v, gamma=gamma, h=h,
        alpha=alpha, theta=theta, q=q,
        throttle=throttle, elevator=elevator, q_max=q_max,
        cost=result.fun, solve_time=solve_time,
        success=result.success, exitflag=result.status,
    )


def save_df_trajectory_result(result: DFTrajectoryResult, path: str | Path) -> None:
    """Save DFTrajectoryResult to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=result.time, V=result.V, gamma=result.gamma, h=result.h,
        alpha=result.alpha, theta=result.theta, q=result.q,
        throttle=result.throttle, elevator=result.elevator, q_max=result.q_max,
        cost=result.cost, solve_time=result.solve_time,
        success=result.success, exitflag=result.exitflag,
    )


def load_df_trajectory_result(path: str | Path) -> DFTrajectoryResult:
    """Load DFTrajectoryResult from .npz file."""
    data = np.load(str(path))
    return DFTrajectoryResult(
        time=data["time"], V=data["V"], gamma=data["gamma"], h=data["h"],
        alpha=data["alpha"], theta=data["theta"], q=data["q"],
        throttle=data["throttle"], elevator=data["elevator"], q_max=data["q_max"],
        cost=float(data["cost"]), solve_time=float(data["solve_time"]),
        success=bool(data["success"]), exitflag=int(data["exitflag"]),
    )
