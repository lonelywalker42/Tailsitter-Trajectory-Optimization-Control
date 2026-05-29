"""Differential flatness transfer for tailsitter UAV longitudinal dynamics.

Refactored from matlab/analysis/controller/DifferentialTransfer.m — given
(V, Vdot, Vddot, Vdddot, gamma, gammadot, gammaddot, gammaddddot), solves
for alpha via Newton iteration, then computes theta, T, Tdot, Tddot, q, qdot,
and elevator deflection.

Also includes:
- Sinusoidal aero coefficient fitting (sin4 model) for analytical derivatives
- Max feasible pitch rate calculation (from calculateMaxFeasiblePitchRate.m)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from .trim import (
    R2D, G, RHO, S_REF, C_REF, M_TOTAL, MS, PROP_R,
    AeroData, load_aero_data,
)


# ---------------------------------------------------------------------------
# Sinusoidal aero coefficient model (sin4)
# ---------------------------------------------------------------------------

@dataclass
class AeroSin4Coeffs:
    """Sinusoidal aero model coefficients (sin4 fit: sum of 4 a*sin(b*x+c) terms).

    Each coefficient array is flat: [a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4].
    The model evaluates: f(alpha_deg) = sum_i a_i * sin(b_i * alpha_deg + c_i).
    """
    cz: np.ndarray   # cz = cD*sin(alpha) + cL*cos(alpha)
    cL: np.ndarray
    cD: np.ndarray
    cm: np.ndarray


def _sin4_model(x: np.ndarray, *params) -> np.ndarray:
    """Evaluate sum of 4 sinusoids: sum_i a_i * sin(b_i * x + c_i)."""
    result = np.zeros_like(x)
    for i in range(4):
        a = params[3 * i]
        b = params[3 * i + 1]
        c = params[3 * i + 2]
        result += a * np.sin(b * x + c)
    return result


def fit_sin4(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a sin4 model to data (x, y). Returns 12 coefficients [a1,b1,c1,...]."""
    # Initial guess: moderate amplitudes, frequencies around 1, zero phase
    p0 = np.zeros(12)
    p0[0::3] = np.std(y)  # amplitudes
    p0[1::3] = [1.0, 2.0, 3.0, 4.0]  # frequencies
    try:
        popt, _ = curve_fit(_sin4_model, x, y, p0=p0, maxfev=10000)
        return popt
    except RuntimeError:
        # Fallback: use Fourier-like initial guess
        p0[1::3] = [0.5, 1.0, 1.5, 2.0]
        popt, _ = curve_fit(_sin4_model, x, y, p0=p0, maxfev=20000)
        return popt


def build_sin4_coeffs(aero: AeroData) -> AeroSin4Coeffs:
    """Build sin4 fit coefficients from aero lookup tables.

    Replicates the logic in matlab/analysis/controller/aero_fit.m:
    cz = cD*sin(alpha) + cL*cos(alpha), then fit sin4 to cz, cL, cD, cm.
    """
    alpha_deg = aero.alpha
    cz = aero.cD * np.sin(np.radians(alpha_deg)) + aero.cL * np.cos(np.radians(alpha_deg))

    return AeroSin4Coeffs(
        cz=fit_sin4(alpha_deg, cz),
        cL=fit_sin4(alpha_deg, aero.cL),
        cD=fit_sin4(alpha_deg, aero.cD),
        cm=fit_sin4(alpha_deg, aero.cm),
    )


def load_sin4_coeffs(config_name: str = "aero_cfg2",
                      data_dir: str | Path = "data/processed") -> AeroSin4Coeffs:
    """Load pre-computed sin4 coefficients from .npy files, or build from aero data.

    If aerofit_{config_name}_*.npy files exist, loads them directly.
    Otherwise, builds from aero lookup tables via fit_sin4.
    """
    d = Path(data_dir)
    prefix = f"aerofit_{config_name}_"

    cz_path = d / f"{prefix}czfit.npy"
    cL_path = d / f"{prefix}cLfit.npy"
    cD_path = d / f"{prefix}cDfit.npy"
    cm_path = d / f"{prefix}cmfit.npy"

    if all(p.exists() for p in [cz_path, cL_path, cD_path, cm_path]):
        return AeroSin4Coeffs(
            cz=np.load(str(cz_path)).flatten(),
            cL=np.load(str(cL_path)).flatten(),
            cD=np.load(str(cD_path)).flatten(),
            cm=np.load(str(cm_path)).flatten(),
        )

    # Build from lookup tables
    aero = load_aero_data(config_name, data_dir)
    return build_sin4_coeffs(aero)


def aero_sin4_eval(alpha_rad: float, coeffs: np.ndarray) -> tuple[float, float, float]:
    """Evaluate sin4 model and its 1st/2nd derivatives w.r.t. alpha in radians.

    Matches matlab/analysis/controller/aero_fit.m exactly:
    - alpha is converted to degrees inside the function
    - derivatives include *57.3 chain rule factor

    Args:
        alpha_rad: Angle of attack in radians
        coeffs: Flat array [a1,b1,c1, a2,b2,c2, a3,b3,c3, a4,b4,c4]

    Returns:
        (fval, fval_dot, fval_ddot) where derivatives are w.r.t. alpha in radians
    """
    alpha_deg = alpha_rad * R2D
    n = len(coeffs) // 3

    fval = 0.0
    fval_dot = 0.0
    fval_ddot = 0.0

    for i in range(n):
        a = coeffs[3 * i]
        b = coeffs[3 * i + 1]
        c = coeffs[3 * i + 2]
        fval += a * np.sin(b * alpha_deg + c)
        # Chain rule: d/d(alpha_rad) = d/d(alpha_deg) * d(alpha_deg)/d(alpha_rad) = *57.3
        fval_dot += a * b * np.cos(b * alpha_deg + c) * R2D
        fval_ddot += -a * b * b * np.sin(b * alpha_deg + c) * R2D**2

    return fval, fval_dot, fval_ddot


# ---------------------------------------------------------------------------
# Differential flatness transfer result
# ---------------------------------------------------------------------------

@dataclass
class DFTransferResult:
    """Result of differential flatness transfer."""
    alpha: float        # angle of attack [rad]
    theta: float        # pitch angle [rad]
    q: float            # pitch rate [rad/s]
    T: float            # thrust [N]
    Tdot: float         # thrust rate [N/s]
    Tddot: float        # thrust acceleration [N/s^2]
    qdot: float         # pitch rate derivative [rad/s^2]
    elevator: float     # elevator deflection [deg]
    residual: float     # Newton iteration residual (for feasibility check)


# ---------------------------------------------------------------------------
# Differential flatness transfer
# ---------------------------------------------------------------------------

def differential_flatness_transfer(
    V: float,
    Vdot: float,
    Vddot: float,
    Vdddot: float,
    gamma: float,
    gammadot: float,
    gammaddot: float,
    gammaddddot: float,
    sin4_coeffs: AeroSin4Coeffs,
    aero: AeroData,
    m: float = M_TOTAL,
    g: float = G,
    rho: float = RHO,
    S: float = S_REF,
    c_bar: float = C_REF,
    Jy: float = 0.183,
    prop_R: float = PROP_R,
    alpha_init: float = 0.0,
    tolerance: float = 2e-3,
    max_iter: int = 50,
) -> DFTransferResult:
    """Differential flatness transfer.

    Given flat outputs (V, gamma) and their derivatives up to 3rd order,
    solves for the full state (alpha, theta, q) and control (T, elevator).

    Ported from matlab/analysis/controller/DifferentialTransfer.m.

    Args:
        V, Vdot, Vddot, Vdddot: Velocity and its derivatives [m/s, m/s^2, m/s^3, m/s^4]
        gamma, gammadot, gammaddot, gammaddddot: Flight path angle and derivatives [rad, rad/s, ...]
        sin4_coeffs: Sinusoidal aero coefficient model
        aero: Aero lookup tables (for elevator inverse lookup)
        m, g, rho, S, c_bar, Jy, prop_R: Physical parameters
        alpha_init: Initial guess for Newton iteration [rad]
        tolerance: Convergence tolerance for Newton iteration
        max_iter: Maximum Newton iterations

    Returns:
        DFTransferResult with all state and control variables
    """
    # Dynamic pressure
    q_dyn = 0.5 * rho * V**2 * S
    q_dyn_dot = rho * S * V * Vdot
    q_dyn_ddot = rho * S * (V * Vddot + Vdot**2)

    # --- Step 1: Solve for alpha via Newton iteration ---
    # Equation: f(alpha) = m*(Vdot*sin(a) - V*gdot*cos(a) + g*sin(gamma)*sin(a) - g*cos(gamma)*cos(a)) + q_dyn*cz(a)

    if abs(V) < 0.01:
        alpha_df = 0.0
        f_current = 0.0
    else:
        alpha_current = alpha_init
        alpha_step = 0.0
        f_current = 0.0

        for _ in range(max_iter):
            sa = np.sin(alpha_current)
            ca = np.cos(alpha_current)

            cz_val, cz_dot, _ = aero_sin4_eval(alpha_current, sin4_coeffs.cz)

            f_current = (m * (Vdot * sa - V * gammadot * ca
                              + g * np.sin(gamma) * sa - g * np.cos(gamma) * ca)
                         + q_dyn * cz_val)

            df_current = (m * (Vdot * ca + V * gammadot * sa
                               + g * np.sin(gamma) * ca + g * np.cos(gamma) * sa)
                          + q_dyn * cz_dot)

            if abs(df_current) < 0.01:
                break

            alpha_step = f_current / df_current
            alpha_current = alpha_current - alpha_step

            if abs(f_current) < tolerance and abs(alpha_step) < tolerance:
                break

        alpha_df = np.clip(alpha_current, -np.pi, np.pi)

    # --- Step 2: Compute aero coefficients at alpha ---
    cD_val, cD_dot, cD_ddot = aero_sin4_eval(alpha_df, sin4_coeffs.cD)
    cL_val, cL_dot, cL_ddot = aero_sin4_eval(alpha_df, sin4_coeffs.cL)

    # --- Step 3: theta = gamma + alpha ---
    theta_df = gamma + alpha_df

    # --- Step 4: Thrust T from force equilibrium ---
    sa = np.sin(alpha_df)
    ca = np.cos(alpha_df)

    if abs(alpha_df - np.pi / 2) < 0.01:
        # Near 90 deg: use lateral equation
        T_df = (m * V * gammadot - q_dyn * cL_val + m * g * np.cos(gamma)) / sa
    else:
        T_df = (m * Vdot + q_dyn * cD_val + m * g * np.sin(gamma)) / ca

    # --- Step 5: q and Tdot from 2x2 linear system ---
    dD = q_dyn_dot * cD_val - q_dyn * cD_dot * gammadot
    dL = q_dyn_dot * cL_val - q_dyn * cL_dot * gammadot

    A1 = np.array([
        [sa * T_df + q_dyn * cD_dot, -ca],
        [-ca * T_df - q_dyn * cL_dot, -sa],
    ])
    B1 = np.array([
        -m * Vddot + T_df * sa * gammadot - m * g * np.cos(gamma) * gammadot - dD,
        -m * (Vdot * gammadot + V * gammaddot) - T_df * ca * gammadot + m * g * np.sin(gamma) * gammadot + dL,
    ])

    sol1 = np.linalg.solve(A1, B1)
    q_df = sol1[0]
    Tdot_df = sol1[1]

    # --- Step 6: qdot and Tddot from another 2x2 system ---
    ddD = (q_dyn_ddot * cD_val
           + 2 * q_dyn_dot * cD_dot * (q_df - gammadot)
           + q_dyn * (cD_ddot * (q_df - gammadot)**2 - cD_dot * gammaddot))
    ddL = (q_dyn_ddot * cL_val
           + 2 * q_dyn_dot * cL_dot * (q_df - gammadot)
           + q_dyn * (cL_ddot * (q_df - gammadot)**2 - cL_dot * gammaddot))

    A2_block = np.array([
        [ca, sa],
        [sa, -ca],
    ])
    B1_block = (q_df - gammadot) * np.array([T_df * q_df, Tdot_df])
    B2_block = np.array([Tdot_df * q_df, 0.0])

    B3 = np.array([
        -m * Vdddot - m * g * np.cos(gamma) * gammaddot + m * g * np.sin(gamma) * gammadot**2
        + Tdot_df * sa * gammadot + T_df * sa * gammaddot + T_df * ca * (q_df * gammadot - gammadot**2) - ddD,
        -m * (Vddot * gammadot + V * gammaddot + 2 * Vdot * gammaddot)
        + m * g * (np.cos(gamma) * gammadot**2 + np.sin(gamma) * gammaddot)
        - Tdot_df * ca * gammadot - T_df * ca * gammaddot
        + T_df * sa * (q_df * gammadot - gammadot**2) + ddL,
    ])

    A4_block = np.array([
        [q_dyn * cD_dot, 0.0],
        [-q_dyn * cL_dot, 0.0],
    ])

    Atotal = A2_block @ np.diag([T_df, 1.0]) + A4_block
    Btotal = B3 - A2_block @ B1_block - A2_block @ B2_block

    sol2 = np.linalg.solve(Atotal, Btotal)
    qdot_df = sol2[0]
    Tddot_df = sol2[1]

    # --- Step 7: Elevator from external moment ---
    cm_val, _, _ = aero_sin4_eval(alpha_df, sin4_coeffs.cm)
    Myext_df = Jy * qdot_df - q_dyn * c_bar * cm_val

    # Induced velocity model
    v_induced = np.sqrt(max(T_df, 0.0) / (2 * rho * np.pi * prop_R**2))
    v_squ = (v_induced + V * ca)**2
    dcm_d = Myext_df / (0.5 * rho * S * c_bar * v_squ) if v_squ > 1e-10 else 0.0

    # Inverse lookup: dcm -> elevator
    ele_d = float(np.interp(dcm_d, aero.dcm, aero.ele))

    return DFTransferResult(
        alpha=alpha_df,
        theta=theta_df,
        q=q_df,
        T=T_df,
        Tdot=Tdot_df,
        Tddot=Tddot_df,
        qdot=qdot_df,
        elevator=ele_d,
        residual=f_current,
    )


# ---------------------------------------------------------------------------
# Max feasible pitch rate
# ---------------------------------------------------------------------------

@dataclass
class PitchRateMarginParams:
    """Tuning parameters for max feasible pitch rate calculation."""
    k: float = 2.0              # proportionality coefficient
    tau_min: float = 0.05       # minimum correction time constant [s]
    tau_max: float = 2.0        # maximum correction time constant [s]
    epsilon_V: float = 0.1      # small number for numerical stability
    epsilon_gamma: float = 0.1
    delta_V: float = 1.0        # velocity characteristic change [m/s]
    delta_gamma: float = 0.1    # gamma characteristic change [rad]
    Cm_delta_e: float = 0.01    # elevator effectiveness derivative [/deg]


@dataclass
class PitchRateMargin:
    """Result of max feasible pitch rate calculation."""
    q_max: float        # max feasible pitch rate [rad/s]
    tau: float          # correction time constant [s]
    tau_V: float        # velocity characteristic time [s]
    tau_gamma: float    # flight path angle characteristic time [s]
    eta: float          # control margin [deg]
    M_delta_e: float    # control effectiveness [N*m]


def calculate_max_feasible_pitch_rate(
    V: float,
    gamma: float,
    V_dot: float,
    gamma_dot: float,
    delta_e_flat: float,
    T_flat: float,
    alpha_flat: float,
    limit: tuple[float, float] = (-20.0, 30.0),
    rho: float = RHO,
    S: float = S_REF,
    c_bar: float = C_REF,
    Jy: float = 0.183,
    prop_R: float = PROP_R,
    params: Optional[PitchRateMarginParams] = None,
) -> PitchRateMargin:
    """Calculate max feasible pitch rate for a given operating point.

    Ported from matlab/analysis/controller/df_trim.m (calculateMaxFeasiblePitchRate).

    Args:
        V: Airspeed [m/s]
        gamma: Flight path angle [rad]
        V_dot: Velocity derivative [m/s^2]
        gamma_dot: FPA derivative [rad/s]
        delta_e_flat: Elevator from flatness [deg]
        T_flat: Thrust from flatness [N]
        alpha_flat: AoA from flatness [rad]
        limit: Elevator limits [min_deg, max_deg]
        rho, S, c_bar, Jy, prop_R: Physical parameters
        params: Tuning parameters (uses defaults if None)

    Returns:
        PitchRateMargin with q_max and intermediate values
    """
    if params is None:
        params = PitchRateMarginParams()

    delta_e_min, delta_e_max = limit

    # Step 1: Characteristic times
    tau_V = params.delta_V / (abs(V_dot) + params.epsilon_V)
    tau_gamma = params.delta_gamma / (abs(gamma_dot) + params.epsilon_gamma)
    tau_prime = min(tau_V, tau_gamma)

    # Step 2: Correction time constant
    tau_unbounded = tau_prime / params.k
    tau = max(params.tau_min, min(params.tau_max, tau_unbounded))

    # Step 3: Control margin
    margin_to_min = delta_e_flat - delta_e_min
    margin_to_max = delta_e_max - delta_e_flat
    eta = min(margin_to_min, margin_to_max)

    # Step 4: Control effectiveness
    v_induced = np.sqrt(max(T_flat, 0.0) / (2 * rho * np.pi * prop_R**2))
    v_squ = (v_induced + V * np.cos(alpha_flat))**2
    dynamic_pressure = 0.5 * rho * v_squ
    M_delta_e = dynamic_pressure * S * c_bar * abs(params.Cm_delta_e)

    # Step 5: Max feasible pitch rate
    q_max = tau * (M_delta_e * eta) / Jy

    return PitchRateMargin(
        q_max=q_max,
        tau=tau,
        tau_V=tau_V,
        tau_gamma=tau_gamma,
        eta=eta,
        M_delta_e=M_delta_e,
    )
