"""Tailsitter UAV trim (equilibrium) analysis.

Refactored from matlab/trim/ — finds steady-state control inputs and
aerodynamic states that satisfy force/moment equilibrium at each (V, theta)
operating point across the flight envelope.

Produces the "transition corridor" showing achievable flight conditions.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AeroData:
    """Aerodynamic lookup tables for trim analysis."""
    # Aero coefficients vs alpha (degrees)
    alpha: np.ndarray
    cL: np.ndarray
    cD: np.ndarray
    cm: np.ndarray
    # Elevator increments vs elevator deflection (degrees)
    ele: np.ndarray
    dcL: np.ndarray
    dcD: np.ndarray
    dcm: np.ndarray
    # Propulsion: throttle -> thrust (kg)
    throttle: np.ndarray
    thrust: np.ndarray


@dataclass
class TrimPoint:
    """Result of a single trim computation."""
    trim_flag: bool = False
    throttle: float = 0.0
    elevator_deg: float = 0.0
    alpha_deg: float = 0.0
    gamma_deg: float = 0.0
    residual: float = 0.0
    residual_moment: float = 0.0


@dataclass
class TrimMesh:
    """Results of a corridor sweep over (V, theta) grid."""
    trim_V: np.ndarray = field(default_factory=lambda: np.array([]))
    trim_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    trim_flag: np.ndarray = field(default_factory=lambda: np.array([]))
    throttle: np.ndarray = field(default_factory=lambda: np.array([]))
    elevator: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    max_moment_ft: np.ndarray = field(default_factory=lambda: np.array([]))
    max_moment_bt: np.ndarray = field(default_factory=lambda: np.array([]))
    # Linearization data (populated by sweep_linearization)
    real_max: Optional[np.ndarray] = None
    real_min: Optional[np.ndarray] = None
    real_num: Optional[np.ndarray] = None
    complex_num: Optional[np.ndarray] = None
    controllability_index: Optional[np.ndarray] = None
    min_singular_value: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aero_data(config_name: str = "aero_cfg2",
                   data_dir: str | Path = "data/processed") -> AeroData:
    """Load aerodynamic lookup tables from .npy files.

    Args:
        config_name: Aero configuration name (e.g., 'aero_cfg1', 'aero_cfg2')
        data_dir: Path to processed data directory

    Returns:
        AeroData with all lookup tables
    """
    d = Path(data_dir)

    def _load(name: str) -> np.ndarray:
        return np.load(d / name).flatten().astype(np.float64)

    return AeroData(
        alpha=_load(f"{config_name}_alpha.npy"),
        cL=_load(f"{config_name}_cL.npy"),
        cD=_load(f"{config_name}_cD.npy"),
        cm=_load(f"{config_name}_cm.npy"),
        ele=_load("aerosurface_ele.npy"),
        dcL=_load("aerosurface_dcL.npy"),
        dcD=_load("aerosurface_dcD.npy"),
        dcm=_load("aerosurface_dcm.npy"),
        throttle=_load("prop_throttle.npy"),
        thrust=_load("prop_thrust.npy"),
    )


# ---------------------------------------------------------------------------
# Physical constants (matching MATLAB Main.m defaults)
# ---------------------------------------------------------------------------

R2D = 180.0 / np.pi
G = 9.8  # matches MATLAB code (not 9.81)
RHO = 1.225
S_REF = 0.62
C_REF = 0.31
M_TOTAL = 6.5
MS = 0.2
PROP_R = 0.2032


# ---------------------------------------------------------------------------
# Aerodynamic computation
# ---------------------------------------------------------------------------

def compute_aero(V: float, alpha_deg: float, ele_deg: float,
                 throttle_cmd: float, aero: AeroData):
    """Compute aerodynamic forces and propulsion at a single operating point.

    Args:
        V: Airspeed (m/s)
        alpha_deg: Angle of attack (degrees)
        ele_deg: Elevator deflection (degrees) — positive = trailing edge down
        throttle_cmd: Throttle command [0, 1]
        aero: Lookup tables

    Returns:
        (L, D, M, Fprop) in N and N·m
    """
    # Propulsion
    Fprop = G * np.interp(throttle_cmd, aero.throttle, aero.thrust)

    # Induced velocity (momentum theory)
    v_induced = np.sqrt(max(Fprop / (2 * RHO * np.pi * PROP_R**2), 0.0))

    # Dynamic pressures
    q = 0.5 * RHO * V**2
    q_induced = 0.5 * RHO * (v_induced + V * np.cos(np.radians(alpha_deg)))**2

    # Aero coefficients
    cL = np.interp(alpha_deg, aero.alpha, aero.cL)
    cD = np.interp(alpha_deg, aero.alpha, aero.cD)
    cm = np.interp(alpha_deg, aero.alpha, aero.cm)

    dcL = np.interp(ele_deg, aero.ele, aero.dcL)
    dcD = np.interp(ele_deg, aero.ele, aero.dcD)
    dcm = np.interp(ele_deg, aero.ele, aero.dcm)

    L = q * S_REF * cL + q_induced * S_REF * dcL
    D = q * S_REF * cD + q_induced * S_REF * dcD
    M = q * S_REF * C_REF * cm + q_induced * S_REF * C_REF * dcm

    return L, D, M, Fprop


def compute_residuals(x: np.ndarray, aero: AeroData,
                      V: float, theta_rad: float, xs: float) -> np.ndarray:
    """Compute force/moment residuals for trim optimization.

    Matches trim_func.m (simple model: elevator increments only in M).

    Args:
        x: [throttle, ele_deg, alpha_rad]
        aero: Lookup tables
        V: Airspeed (m/s)
        theta_rad: Pitch angle (rad)
        xs: Moving mass position (m)

    Returns:
        3D residual vector [Fx, Fz, M]
    """
    throttle_cmd = x[0]
    ele_deg = -x[1]  # sign inversion: aero data uses negative = trailing edge down
    alpha_rad = x[2]
    alpha_deg = R2D * alpha_rad

    # Propulsion
    Fprop = G * np.interp(throttle_cmd, aero.throttle, aero.thrust)

    # Induced velocity
    v_induced = np.sqrt(max(Fprop / (2 * RHO * np.pi * PROP_R**2), 0.0))

    # Dynamic pressures
    q_dyn = 0.5 * RHO * V**2
    q_induced = 0.5 * RHO * (v_induced + V * np.cos(alpha_rad))**2

    # Aero coefficients (body aero only, no elevator increment in L/D)
    cL = np.interp(alpha_deg, aero.alpha, aero.cL)
    cD = np.interp(alpha_deg, aero.alpha, aero.cD)
    cm = np.interp(alpha_deg, aero.alpha, aero.cm)
    dcm = np.interp(ele_deg, aero.ele, aero.dcm)

    L = q_dyn * S_REF * cL
    D = q_dyn * S_REF * cD
    M = q_dyn * S_REF * C_REF * cm + q_induced * S_REF * C_REF * dcm

    # Force/moment residuals
    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)

    temp = np.zeros(3)
    temp[0] = L * sa - D * ca + 2 * Fprop - M_TOTAL * G * st
    temp[1] = -L * ca - D * sa + M_TOTAL * G * ct
    temp[2] = M - MS * G * xs

    return temp


def compute_residual_norm(x: np.ndarray, aero: AeroData,
                          V: float, theta_rad: float, xs: float) -> float:
    """Scalar residual norm for optimization (matches trim_func.m)."""
    temp = compute_residuals(x, aero, V, theta_rad, xs)
    return float(np.sqrt(np.dot(temp, temp)))


def compute_residuals_full(x: np.ndarray, aero: AeroData,
                           V: float, theta_rad: float, xs: float) -> np.ndarray:
    """Compute residuals with elevator increments in L, D, and M.

    Matches trim_neqfunc.m / trim_Theta.m (full model).
    """
    throttle_cmd = x[0]
    ele_deg = -x[1]
    alpha_rad = x[2]
    alpha_deg = R2D * alpha_rad

    Fprop = G * np.interp(throttle_cmd, aero.throttle, aero.thrust)
    v_induced = np.sqrt(max(Fprop / (2 * RHO * np.pi * PROP_R**2), 0.0))

    q_dyn = 0.5 * RHO * V**2
    q_induced = 0.5 * RHO * (v_induced + V * np.cos(alpha_rad))**2

    cL = np.interp(alpha_deg, aero.alpha, aero.cL)
    cD = np.interp(alpha_deg, aero.alpha, aero.cD)
    cm = np.interp(alpha_deg, aero.alpha, aero.cm)
    dcL = np.interp(ele_deg, aero.ele, aero.dcL)
    dcD = np.interp(ele_deg, aero.ele, aero.dcD)
    dcm = np.interp(ele_deg, aero.ele, aero.dcm)

    L = q_dyn * S_REF * cL + q_induced * S_REF * dcL
    D = q_dyn * S_REF * cD + q_induced * S_REF * dcD
    M = q_dyn * S_REF * C_REF * cm + q_induced * S_REF * C_REF * dcm

    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)

    temp = np.zeros(3)
    temp[0] = L * sa - D * ca + 2 * Fprop - M_TOTAL * G * st
    temp[1] = -L * ca - D * sa + M_TOTAL * G * ct
    temp[2] = M - MS * G * xs * ct

    return temp


def compute_residual_moment(x: np.ndarray, aero: AeroData,
                            V: float, theta_rad: float, xs: float,
                            ForB: float) -> float:
    """Signed residual moment for max-moment optimization (matches trimMy.m).

    Args:
        ForB: -1 for forward transition (max Mrest > 0), +1 for backward (min Mrest < 0)
    """
    throttle_cmd = x[0]
    ele_deg = -x[1]
    alpha_rad = x[2]
    alpha_deg = R2D * alpha_rad

    Fprop = G * np.interp(throttle_cmd, aero.throttle, aero.thrust)
    v_induced = np.sqrt(max(Fprop / (2 * RHO * np.pi * PROP_R**2), 0.0))

    q_dyn = 0.5 * RHO * V**2
    q_induced = 0.5 * RHO * (v_induced + V * np.cos(alpha_rad))**2

    cm_val = np.interp(alpha_deg, aero.alpha, aero.cm)
    dcm_val = np.interp(ele_deg, aero.ele, aero.dcm)

    M = q_dyn * S_REF * C_REF * cm_val + q_induced * S_REF * C_REF * dcm_val
    Mrest = M - MS * G * xs

    return ForB * Mrest


def compute_force_constraint(x: np.ndarray, aero: AeroData,
                             V: float, theta_rad: float, xs: float) -> float:
    """Force equilibrium constraint for max-moment optimization (matches trimForce.m).

    Returns value <= 0 when constraint is satisfied.
    """
    throttle_cmd = x[0]
    ele_deg = -x[1]
    alpha_rad = x[2]
    alpha_deg = R2D * alpha_rad

    Fprop = G * np.interp(throttle_cmd, aero.throttle, aero.thrust)
    v_induced = np.sqrt(max(Fprop / (2 * RHO * np.pi * PROP_R**2), 0.0))

    q_dyn = 0.5 * RHO * V**2

    cL = np.interp(alpha_deg, aero.alpha, aero.cL)
    cD = np.interp(alpha_deg, aero.alpha, aero.cD)

    L = q_dyn * S_REF * cL
    D = q_dyn * S_REF * cD

    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)

    temp = np.zeros(2)
    temp[0] = L * sa - D * ca + 2 * Fprop - M_TOTAL * G * st
    temp[1] = -L * ca - D * sa + M_TOTAL * G * ct

    eps = 1e-3
    return float(np.sqrt(np.dot(temp, temp))) - eps


# ---------------------------------------------------------------------------
# Trim solvers
# ---------------------------------------------------------------------------

def trim_level_flight(V: float, aero: AeroData, xs: float = 0.0) -> TrimPoint:
    """Find level flight trim (theta=alpha, gamma=0).

    Maps trim_Theta.m. In level flight, the flight path angle is zero,
    so theta = alpha.

    Args:
        V: Airspeed (m/s)
        aero: Lookup tables
        xs: Moving mass position (m)

    Returns:
        TrimPoint with trim results
    """
    x0 = np.array([0.65, 0.0, np.pi / 12])
    lb = np.array([0.0, -30.0, -np.pi / 3])
    if V > 10:
        ub = np.array([1.0, 20.0, 16 * np.pi / 180])
    else:
        ub = np.array([1.0, 20.0, np.pi / 2])

    theta_rad = x0[2]  # Will be updated during optimization

    def objective(x):
        # In level flight: theta = alpha
        alpha_rad = x[2]
        theta_rad_local = alpha_rad
        res = compute_residuals_full(x, aero, V, theta_rad_local, xs)
        return float(np.dot(res, res))

    result = minimize(objective, x0, method='SLSQP',
                      bounds=list(zip(lb, ub)),
                      options={'disp': False, 'ftol': 1e-10, 'maxiter': 500})

    eps = 1e-2
    tp = TrimPoint()
    if result.fun < eps:
        tp.trim_flag = True
        tp.throttle = result.x[0]
        tp.elevator_deg = result.x[1]
        tp.alpha_deg = R2D * result.x[2]
        tp.gamma_deg = 0.0  # level flight
        tp.residual = np.sqrt(result.fun)
    return tp


def trim_at_conditions(V: float, theta_deg: float, aero: AeroData,
                       xs: float = 0.0, max_restarts: int = 15) -> TrimPoint:
    """Find trim at given (V, theta) with random restarts.

    Maps trim_Alpha.m. Searches for alpha within ±45° of theta.

    Args:
        V: Airspeed (m/s)
        theta_deg: Pitch angle (degrees)
        aero: Lookup tables
        xs: Moving mass position (m)
        max_restarts: Maximum number of random restarts

    Returns:
        TrimPoint with trim results
    """
    theta_rad = theta_deg / R2D

    lb = np.array([0.0, -30.0, theta_rad - 45 * np.pi / 180])
    ub = np.array([1.0, 20.0, theta_rad + 45 * np.pi / 180])

    eps = 1e-3
    best_tp = TrimPoint()
    best_fval = np.inf

    rng = np.random.default_rng()

    for _ in range(max_restarts):
        x0 = lb + (ub - lb) * rng.random(3)

        def objective(x, _V=V, _theta=theta_rad, _xs=xs):
            return compute_residual_norm(x, aero, _V, _theta, _xs)

        result = minimize(objective, x0, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          options={'disp': False, 'ftol': 1e-10, 'maxiter': 500})

        if result.fun < best_fval:
            best_fval = result.fun
            if result.fun < eps:
                best_tp = TrimPoint(
                    trim_flag=True,
                    throttle=result.x[0],
                    elevator_deg=result.x[1],
                    alpha_deg=R2D * result.x[2],
                    gamma_deg=theta_deg - R2D * result.x[2],
                    residual=np.sqrt(result.fun),
                )
            if result.fun < eps:
                break

    return best_tp


def trim_max_moment(V: float, theta_deg: float, aero: AeroData,
                    xs: float = 0.0, ForB: float = -1,
                    max_restarts: int = 15) -> TrimPoint:
    """Find trim that maximizes residual moment subject to force equilibrium.

    Maps trim_MaxMy.m. Uses force constraint (norm < eps) and minimizes
    signed residual moment.

    Args:
        V: Airspeed (m/s)
        theta_deg: Pitch angle (degrees)
        aero: Lookup tables
        xs: Moving mass position (m)
        ForB: -1 for forward transition, +1 for backward transition
        max_restarts: Maximum number of random restarts

    Returns:
        TrimPoint with trim results and residual_moment
    """
    theta_rad = theta_deg / R2D

    lb = np.array([0.0, -30.0, theta_rad - 45 * np.pi / 180])
    ub = np.array([1.0, 20.0, theta_rad + 45 * np.pi / 180])

    best_tp = TrimPoint()
    best_exitflag = -1

    rng = np.random.default_rng()

    for _ in range(max_restarts):
        x0 = lb + (ub - lb) * rng.random(3)

        def objective(x):
            return compute_residual_moment(x, aero, V, theta_rad, xs, ForB)

        constraints = [{
            'type': 'ineq',
            'fun': lambda x: -compute_force_constraint(x, aero, V, theta_rad, xs)
        }]

        result = minimize(objective, x0, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints,
                          options={'disp': False, 'ftol': 1e-10, 'maxiter': 500})

        if result.success and ForB * result.fun < 0:
            best_tp = TrimPoint(
                trim_flag=True,
                throttle=result.x[0],
                elevator_deg=result.x[1],
                alpha_deg=R2D * result.x[2],
                gamma_deg=theta_deg - R2D * result.x[2],
                residual_moment=ForB * result.fun,
            )
            break

    return best_tp


# ---------------------------------------------------------------------------
# Corridor sweep functions
# ---------------------------------------------------------------------------

def _sweep_single_point(V: float, theta_deg: float, aero: AeroData,
                        xs: float, max_restarts: int) -> dict:
    """Process a single (V, theta) point for corridor sweep.

    Returns dict with trim results for equilibrium and max moment.
    """
    # Equilibrium trim
    tp = trim_at_conditions(V, theta_deg, aero, xs, max_restarts)

    # Max moment for forward and backward transition
    tp_ft = trim_max_moment(V, theta_deg, aero, xs, ForB=-1, max_restarts=15)
    tp_bt = trim_max_moment(V, theta_deg, aero, xs, ForB=1, max_restarts=15)

    # Combined flag: at least one of FT/BT must succeed
    flag = min(tp.trim_flag + (tp_ft.trim_flag or tp_bt.trim_flag), 1)

    return {
        'flag': flag if tp.trim_flag else 0,
        'throttle': tp.throttle,
        'elevator': tp.elevator_deg,
        'alpha': tp.alpha_deg,
        'gamma': tp.gamma_deg,
        'moment_ft': tp_ft.residual_moment if tp_ft.trim_flag else 0.0,
        'moment_bt': tp_bt.residual_moment if tp_bt.trim_flag else 0.0,
    }


def sweep_level_flight(V_range: np.ndarray, aero: AeroData,
                       xs: float = 0.0) -> dict:
    """Sweep level flight trim over velocity range.

    Maps trimLF.m.

    Args:
        V_range: Array of airspeed values (m/s)
        aero: Lookup tables
        xs: Moving mass position (m)

    Returns:
        Dict with arrays: V, flag, throttle, elevator, alpha, theta
    """
    n = len(V_range)
    result = {
        'V': V_range.copy(),
        'flag': np.zeros(n),
        'throttle': np.zeros(n),
        'elevator': np.zeros(n),
        'alpha': np.zeros(n),
        'theta': np.zeros(n),
    }

    for i, V in enumerate(V_range):
        tp = trim_level_flight(V, aero, xs)
        result['flag'][i] = tp.trim_flag
        result['throttle'][i] = tp.throttle
        result['elevator'][i] = tp.elevator_deg
        result['alpha'][i] = tp.alpha_deg
        result['theta'][i] = tp.alpha_deg  # theta = alpha in level flight

        if tp.trim_flag:
            print(f"V={V:.1f} m/s: throttle={tp.throttle:.3f}, "
                  f"de={tp.elevator_deg:.2f} deg, alpha={tp.alpha_deg:.2f} deg")
        else:
            print(f"V={V:.1f} m/s: trim failed")

    return result


def sweep_corridor(V_range: np.ndarray, theta_range: np.ndarray,
                   aero: AeroData, xs: float = 0.0,
                   max_restarts: int = 15,
                   n_jobs: int = 1) -> TrimMesh:
    """Sweep transition corridor over (V, theta) grid.

    Maps trimCorPar.m. Uses parallel processing.

    Args:
        V_range: Array of airspeed values (m/s)
        theta_range: Array of pitch angle values (degrees)
        aero: Lookup tables
        xs: Moving mass position (m)
        max_restarts: Maximum optimizer restarts per point
        n_jobs: Number of parallel workers (1 = sequential)

    Returns:
        TrimMesh with all results
    """
    n_v = len(V_range)
    n_theta = len(theta_range)
    total = n_v * n_theta

    mesh = TrimMesh(
        trim_V=V_range.copy(),
        trim_theta=theta_range.copy(),
        trim_flag=np.zeros((n_v, n_theta)),
        throttle=np.zeros((n_v, n_theta)),
        elevator=np.zeros((n_v, n_theta)),
        alpha=np.zeros((n_v, n_theta)),
        gamma=np.zeros((n_v, n_theta)),
        max_moment_ft=np.zeros((n_v, n_theta)),
        max_moment_bt=np.zeros((n_v, n_theta)),
    )

    # Build task list
    tasks = []
    for i in range(n_v):
        for j in range(n_theta):
            tasks.append((i, j, V_range[i], theta_range[j]))

    if n_jobs <= 1:
        # Sequential execution
        for idx, (i, j, V, theta) in enumerate(tasks):
            result = _sweep_single_point(V, theta, aero, xs, max_restarts)
            _store_result(mesh, i, j, result)
            print(f"[{idx+1}/{total}] V={V:.1f}, theta={theta:.1f}: "
                  f"flag={result['flag']}")
    else:
        # Parallel execution
        # Note: aero data is pickled for multiprocessing
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, j, V, theta in tasks:
                future = executor.submit(
                    _sweep_single_point, V, theta, aero, xs, max_restarts)
                futures[future] = (i, j, V, theta)

            for idx, future in enumerate(as_completed(futures)):
                i, j, V, theta = futures[future]
                result = future.result()
                _store_result(mesh, i, j, result)
                print(f"[{idx+1}/{total}] V={V:.1f}, theta={theta:.1f}: "
                      f"flag={result['flag']}")

    return mesh


def _store_result(mesh: TrimMesh, i: int, j: int, result: dict):
    """Store a single point result into the mesh arrays."""
    mesh.trim_flag[i, j] = result['flag']
    mesh.throttle[i, j] = result['throttle']
    mesh.elevator[i, j] = result['elevator']
    mesh.alpha[i, j] = result['alpha']
    mesh.gamma[i, j] = result['gamma']
    mesh.max_moment_ft[i, j] = result['moment_ft']
    mesh.max_moment_bt[i, j] = result['moment_bt']
