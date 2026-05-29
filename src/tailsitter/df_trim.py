"""Differential-flatness-based trim corridor sweep.

Refactored from matlab/analysis/controller/df_trim.m — sweeps a (V, gamma)
grid, calls differential_flatness_transfer at each point, and computes the
max feasible pitch rate grid.

Unlike the force/moment-equilibrium trim in trim.py, this operates in the
differential flatness (V, gamma) state space with constant Vdot and gammadot.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .differential_flatness import (
    AeroSin4Coeffs,
    DFTransferResult,
    PitchRateMargin,
    PitchRateMarginParams,
    calculate_max_feasible_pitch_rate,
    differential_flatness_transfer,
    load_sin4_coeffs,
)
from .trim import (
    R2D, G, RHO, S_REF, C_REF, M_TOTAL, MS, PROP_R,
    AeroData, load_aero_data,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DFTrimPoint:
    """Single point in DF-based trim mesh."""
    alpha: float = 0.0          # angle of attack [rad]
    theta: float = 0.0          # pitch angle [rad]
    q: float = 0.0              # pitch rate [rad/s]
    T: float = 0.0              # thrust [N]
    Tdot: float = 0.0           # thrust rate [N/s]
    Tddot: float = 0.0          # thrust acceleration [N/s^2]
    qdot: float = 0.0           # pitch rate derivative [rad/s^2]
    elevator: float = 0.0       # elevator deflection [deg]
    trim_flag: bool = False     # feasible trim point
    residual: float = 0.0       # Newton residual


@dataclass
class DFTrimMesh:
    """Results of DF-based corridor sweep over (V, gamma) grid."""
    trim_V: np.ndarray = field(default_factory=lambda: np.array([]))
    trim_gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    # Grids are (n_gamma, n_V) — matching MATLAB convention
    V_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    flag_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    q_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    T_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    ele_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    qdot_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    qmax_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    tau_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    eta_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    # Parameters used
    Vdot: float = 0.0
    gammadot: float = 0.0


# ---------------------------------------------------------------------------
# Single-point trim via differential flatness
# ---------------------------------------------------------------------------

def _df_trim_single(
    V: float,
    gamma_deg: float,
    Vdot: float,
    gammadot: float,
    sin4_coeffs: AeroSin4Coeffs,
    aero: AeroData,
    limit: tuple[float, float],
    phys_kwargs: dict,
    margin_params: Optional[PitchRateMarginParams],
) -> DFTrimPoint:
    """Compute DF trim at a single (V, gamma) point."""
    gamma_rad = np.radians(gamma_deg)

    try:
        result = differential_flatness_transfer(
            V=V, Vdot=Vdot, Vddot=0.0, Vdddot=0.0,
            gamma=gamma_rad, gammadot=gammadot, gammaddot=0.0, gammaddddot=0.0,
            sin4_coeffs=sin4_coeffs, aero=aero,
            **phys_kwargs,
        )

        # Feasibility check (matching MATLAB: residual <= 0.002, |alpha| < pi/2, T >= 0)
        trim_flag = (
            abs(result.residual) <= 0.002
            and abs(result.alpha) < np.pi / 2
            and result.T >= 0
        )

        return DFTrimPoint(
            alpha=result.alpha,
            theta=result.theta,
            q=result.q,
            T=result.T,
            Tdot=result.Tdot,
            Tddot=result.Tddot,
            qdot=result.qdot,
            elevator=result.elevator,
            trim_flag=trim_flag,
            residual=result.residual,
        )
    except Exception:
        return DFTrimPoint()


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------

def df_trim_sweep(
    V_range: tuple[float, float] = (0.0, 20.0),
    gamma_range: tuple[float, float] = (0.0, 90.0),
    dV: float = 0.5,
    dgamma: float = 1.0,
    Vdot: float = 0.0,
    gammadot: float = 0.0,
    config_name: str = "aero_cfg2",
    data_dir: str | Path = "data/processed",
    limit: tuple[float, float] = (-20.0, 30.0),
    margin_params: Optional[PitchRateMarginParams] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> DFTrimMesh:
    """Sweep a (V, gamma) grid using differential flatness trim.

    Ported from matlab/analysis/controller/df_trim.m.

    Args:
        V_range: (V_min, V_max) in m/s
        gamma_range: (gamma_min, gamma_max) in degrees
        dV: Velocity step [m/s]
        dgamma: FPA step [deg]
        Vdot: Constant velocity derivative [m/s^2]
        gammadot: Constant FPA derivative [rad/s]
        config_name: Aero config name
        data_dir: Path to processed data
        limit: Elevator limits [min_deg, max_deg]
        margin_params: Pitch rate margin tuning parameters
        n_workers: Number of parallel workers (None = auto)
        verbose: Print progress

    Returns:
        DFTrimMesh with all grid data
    """
    # Load data
    aero = load_aero_data(config_name, data_dir)
    sin4_coeffs = load_sin4_coeffs(config_name, data_dir)

    # Build grid
    trim_V = np.arange(V_range[0], V_range[1] + dV * 0.5, dV)
    trim_gamma = np.arange(gamma_range[0], gamma_range[1] + dgamma * 0.5, dgamma)
    n_V = len(trim_V)
    n_gamma = len(trim_gamma)
    total = n_V * n_gamma

    phys_kwargs = dict(
        m=M_TOTAL, g=G, rho=RHO, S=S_REF, c_bar=C_REF,
        Jy=0.183, prop_R=PROP_R,
    )

    if verbose:
        print(f"DF trim sweep: {n_V} x {n_gamma} = {total} points")
        print(f"  V: {V_range[0]:.1f} to {V_range[1]:.1f} m/s, dV={dV}")
        print(f"  gamma: {gamma_range[0]:.1f} to {gamma_range[1]:.1f} deg, dgamma={dgamma}")
        print(f"  Vdot={Vdot}, gammadot={gammadot}")

    # Compute all points (sequential for now; can parallelize later)
    points = []
    for j in range(n_gamma):
        for i in range(n_V):
            pt = _df_trim_single(
                trim_V[i], trim_gamma[j], Vdot, gammadot,
                sin4_coeffs, aero, limit, phys_kwargs, margin_params,
            )
            points.append(pt)

    # Reshape to grids (gamma along rows, V along columns)
    flag_grid = np.zeros((n_gamma, n_V))
    alpha_grid = np.zeros((n_gamma, n_V))
    theta_grid = np.zeros((n_gamma, n_V))
    q_grid = np.zeros((n_gamma, n_V))
    T_grid = np.zeros((n_gamma, n_V))
    ele_grid = np.zeros((n_gamma, n_V))
    qdot_grid = np.zeros((n_gamma, n_V))

    for j in range(n_gamma):
        for i in range(n_V):
            pt = points[j * n_V + i]
            flag_grid[j, i] = 1.0 if pt.trim_flag else 0.0
            alpha_grid[j, i] = pt.alpha * R2D
            theta_grid[j, i] = pt.theta * R2D
            q_grid[j, i] = pt.q * R2D
            T_grid[j, i] = pt.T
            ele_grid[j, i] = pt.elevator
            qdot_grid[j, i] = pt.qdot * R2D

    # Build meshgrid
    V_grid, gamma_grid = np.meshgrid(trim_V, trim_gamma)

    # Compute max feasible pitch rate grid
    qmax_grid = np.zeros((n_gamma, n_V))
    tau_grid = np.zeros((n_gamma, n_V))
    eta_grid = np.zeros((n_gamma, n_V))

    for j in range(n_gamma):
        for i in range(n_V):
            if flag_grid[j, i] > 0:
                alpha_rad = np.radians(alpha_grid[j, i])
                margin = calculate_max_feasible_pitch_rate(
                    V=V_grid[j, i],
                    gamma=np.radians(gamma_grid[j, i]),
                    V_dot=Vdot,
                    gamma_dot=gammadot,
                    delta_e_flat=ele_grid[j, i],
                    T_flat=T_grid[j, i],
                    alpha_flat=alpha_rad,
                    limit=limit,
                    params=margin_params,
                )
                qmax_grid[j, i] = np.degrees(margin.q_max)
                tau_grid[j, i] = margin.tau
                eta_grid[j, i] = margin.eta

    if verbose:
        n_feasible = int(np.sum(flag_grid))
        print(f"  Feasible points: {n_feasible}/{total} ({100*n_feasible/total:.1f}%)")

    return DFTrimMesh(
        trim_V=trim_V,
        trim_gamma=trim_gamma,
        V_grid=V_grid,
        gamma_grid=gamma_grid,
        flag_grid=flag_grid,
        alpha_grid=alpha_grid,
        theta_grid=theta_grid,
        q_grid=q_grid,
        T_grid=T_grid,
        ele_grid=ele_grid,
        qdot_grid=qdot_grid,
        qmax_grid=qmax_grid,
        tau_grid=tau_grid,
        eta_grid=eta_grid,
        Vdot=Vdot,
        gammadot=gammadot,
    )


def save_df_trim_mesh(mesh: DFTrimMesh, path: str | Path) -> None:
    """Save DFTrimMesh to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        trim_V=mesh.trim_V,
        trim_gamma=mesh.trim_gamma,
        V_grid=mesh.V_grid,
        gamma_grid=mesh.gamma_grid,
        flag_grid=mesh.flag_grid,
        alpha_grid=mesh.alpha_grid,
        theta_grid=mesh.theta_grid,
        q_grid=mesh.q_grid,
        T_grid=mesh.T_grid,
        ele_grid=mesh.ele_grid,
        qdot_grid=mesh.qdot_grid,
        qmax_grid=mesh.qmax_grid,
        tau_grid=mesh.tau_grid,
        eta_grid=mesh.eta_grid,
        Vdot=mesh.Vdot,
        gammadot=mesh.gammadot,
    )


def load_df_trim_mesh(path: str | Path) -> DFTrimMesh:
    """Load DFTrimMesh from .npz file."""
    data = np.load(str(path))
    return DFTrimMesh(
        trim_V=data["trim_V"],
        trim_gamma=data["trim_gamma"],
        V_grid=data["V_grid"],
        gamma_grid=data["gamma_grid"],
        flag_grid=data["flag_grid"],
        alpha_grid=data["alpha_grid"],
        theta_grid=data["theta_grid"],
        q_grid=data["q_grid"],
        T_grid=data["T_grid"],
        ele_grid=data["ele_grid"],
        qdot_grid=data["qdot_grid"],
        qmax_grid=data["qmax_grid"],
        tau_grid=data["tau_grid"],
        eta_grid=data["eta_grid"],
        Vdot=float(data["Vdot"]),
        gammadot=float(data["gammadot"]),
    )
