"""Linearization and eigenvalue analysis for tailsitter UAV trim points.

Refactored from matlab/trim/linearization/ — computes numerical Jacobians,
eigenvalue/controllability/stability/modal analysis at trim conditions.

State vector for linearization (4D): [V, alpha, theta, q]
Control vector (2D): [throttle, elevator_deg]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import linalg

from .trim import AeroData, TrimPoint, TrimMesh, R2D, G, RHO, S_REF, C_REF, M_TOTAL, MS, PROP_R


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EigenInfo:
    """Eigenvalue analysis results."""
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    real_eigen_count: int = 0
    complex_eigen_count: int = 0
    max_real_part: float = 0.0
    min_real_part: float = 0.0
    magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ControllabilityInfo:
    """Controllability analysis results."""
    rank: int = 0
    is_fully_controllable: bool = False
    controllability_index: float = 0.0
    min_singular_value: float = 0.0
    max_singular_value: float = 0.0
    modal_controllability: np.ndarray = field(default_factory=lambda: np.array([]))
    controllable_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    num_controllable: int = 0


@dataclass
class StabilityInfo:
    """Stability analysis results."""
    type: str = "unknown"
    margin: float = 0.0
    max_real_part: float = 0.0
    has_oscillatory_modes: bool = False
    max_frequency: float = 0.0
    min_damping_ratio: float = float('inf')


@dataclass
class ModalInfo:
    """Single mode analysis result."""
    type: str = "unknown"
    eigenvalue: complex = 0j
    time_constant: float = 0.0
    settling_time: float = 0.0
    natural_frequency: float = 0.0
    damping_ratio: float = 0.0
    damped_frequency: float = 0.0
    overshoot: float = 0.0
    dominant_states: list = field(default_factory=list)


@dataclass
class SystemAnalysis:
    """Complete system analysis at a trim point."""
    eigen_info: EigenInfo = field(default_factory=EigenInfo)
    controllability_info: ControllabilityInfo = field(default_factory=ControllabilityInfo)
    stability_info: StabilityInfo = field(default_factory=StabilityInfo)
    modal_info: list = field(default_factory=list)
    max_q_perturbation: float = 0.0


# ---------------------------------------------------------------------------
# 4-DOF longitudinal dynamics for linearization
# ---------------------------------------------------------------------------

def lon_dynamics(x: np.ndarray, u: np.ndarray, aero: AeroData,
                 xs: float = 0.0) -> np.ndarray:
    """4-DOF longitudinal dynamics for linearization.

    Maps lon_dyn.m. State: [V, alpha, theta, q], Control: [throttle, ele_deg].

    Args:
        x: State [V, alpha_rad, theta_rad, q]
        u: Control [throttle, ele_deg]
        aero: Lookup tables
        xs: Moving mass position

    Returns:
        State derivative [Vdot, alphadot, thetadot, qdot]
    """
    V = x[0]
    alpha_rad = x[1]
    theta_rad = x[2]
    q = x[3]

    throttle = u[0]
    ele_deg = -u[1]  # sign inversion

    alpha_deg = R2D * alpha_rad

    # Propulsion
    Fprop = G * np.interp(throttle, aero.throttle, aero.thrust)

    # Induced velocity
    if Fprop > 0:
        v_induced = np.sqrt(Fprop / (2 * RHO * np.pi * PROP_R**2))
    else:
        v_induced = 0.0

    # Dynamic pressures
    q_dyn = 0.5 * RHO * V**2
    q_induced = 0.5 * RHO * (v_induced + V * np.cos(alpha_rad))**2

    # Aero coefficients
    cL = np.interp(alpha_deg, aero.alpha, aero.cL)
    cD = np.interp(alpha_deg, aero.alpha, aero.cD)
    cm = np.interp(alpha_deg, aero.alpha, aero.cm)
    dcm = np.interp(ele_deg, aero.ele, aero.dcm)

    L = q_dyn * S_REF * cL
    D = q_dyn * S_REF * cD
    M = q_dyn * S_REF * C_REF * cm + q_induced * S_REF * C_REF * dcm

    # State derivatives
    theta_minus_alpha = theta_rad - alpha_rad
    Vdot = (-D - M_TOTAL * G * np.sin(theta_minus_alpha) +
            2 * Fprop * np.cos(alpha_rad)) / M_TOTAL

    if V > 1e-10:
        alphadot = (-L + M_TOTAL * G * np.cos(theta_minus_alpha) -
                    2 * Fprop * np.sin(alpha_rad)) / (M_TOTAL * V) + q
    else:
        alphadot = q

    thetadot = q

    Jy = 0.183  # moment of inertia
    qdot = (M - MS * G * xs) / Jy

    return np.array([Vdot, alphadot, thetadot, qdot])


def linearize_at_trim(trim_point: TrimPoint, aero: AeroData,
                      xs: float = 0.0, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Compute linearized A, B matrices at a trim point via central differences.

    Maps myLinearize.m (without Simulink dependency).

    Args:
        trim_point: Trim results (throttle, elevator, alpha)
        aero: Lookup tables
        xs: Moving mass position
        eps: Perturbation size for finite differences

    Returns:
        (A, B) matrices: A is 4x4, B is 4x2
    """
    # Trim state: [V, alpha, theta, q]
    # At trim, theta = alpha + gamma, q = 0
    alpha_rad = trim_point.alpha_deg / R2D
    gamma_rad = trim_point.gamma_deg / R2D
    theta_rad = alpha_rad + gamma_rad

    # Estimate V from trim conditions (use a reasonable default)
    # This requires knowing V from the sweep context
    # We'll use the throttle to estimate V from force equilibrium
    V_trim = 10.0  # default; caller should override if needed

    x0 = np.array([V_trim, alpha_rad, theta_rad, 0.0])
    u0 = np.array([trim_point.throttle, trim_point.elevator_deg])

    n = 4
    m = 2
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # Central finite differences for A matrix
    for j in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps

        f_plus = lon_dynamics(x_plus, u0, aero, xs)
        f_minus = lon_dynamics(x_minus, u0, aero, xs)
        A[:, j] = (f_plus - f_minus) / (2 * eps)

    # Central finite differences for B matrix
    for j in range(m):
        u_plus = u0.copy()
        u_minus = u0.copy()
        u_plus[j] += eps
        u_minus[j] -= eps

        f_plus = lon_dynamics(x0, u_plus, aero, xs)
        f_minus = lon_dynamics(x0, u_minus, aero, xs)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B


def linearize_at_point(V: float, alpha_deg: float, theta_deg: float,
                       throttle: float, elevator_deg: float,
                       aero: AeroData, xs: float = 0.0,
                       eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Compute linearized A, B at a specific operating point.

    Args:
        V: Airspeed (m/s)
        alpha_deg: Angle of attack (degrees)
        theta_deg: Pitch angle (degrees)
        throttle: Throttle command [0, 1]
        elevator_deg: Elevator deflection (degrees)
        aero: Lookup tables
        xs: Moving mass position
        eps: Perturbation size

    Returns:
        (A, B) matrices
    """
    alpha_rad = alpha_deg / R2D
    theta_rad = theta_deg / R2D

    x0 = np.array([V, alpha_rad, theta_rad, 0.0])
    u0 = np.array([throttle, elevator_deg])

    n = 4
    m = 2
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for j in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        A[:, j] = (lon_dynamics(x_plus, u0, aero, xs) -
                    lon_dynamics(x_minus, u0, aero, xs)) / (2 * eps)

    for j in range(m):
        u_plus = u0.copy()
        u_minus = u0.copy()
        u_plus[j] += eps
        u_minus[j] -= eps
        B[:, j] = (lon_dynamics(x0, u_plus, aero, xs) -
                    lon_dynamics(x0, u_minus, aero, xs)) / (2 * eps)

    return A, B


# ---------------------------------------------------------------------------
# System analysis functions
# ---------------------------------------------------------------------------

def analyze_eigenvalues(A: np.ndarray) -> EigenInfo:
    """Analyze eigenvalues of system matrix A.

    Maps eigenvalueAnalysis.m / analyze_eigenvalues() in LinearSysAnalysis.m.

    Args:
        A: 4x4 system matrix

    Returns:
        EigenInfo with analysis results
    """
    eigenvalues, eigenvectors = linalg.eig(A)

    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    real_mask = np.abs(imag_parts) < 1e-10

    info = EigenInfo(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        real_eigen_count=int(np.sum(real_mask)),
        complex_eigen_count=int(np.sum(~real_mask)),
        max_real_part=float(np.max(real_parts)),
        min_real_part=float(np.min(real_parts)),
        magnitudes=np.abs(eigenvalues),
    )

    return info


def analyze_controllability(A: np.ndarray, B: np.ndarray,
                            eigenvalues: np.ndarray) -> ControllabilityInfo:
    """Analyze system controllability.

    Maps analyze_controllability() in LinearSysAnalysis.m.

    Args:
        A: 4x4 system matrix
        B: 4x2 input matrix
        eigenvalues: Pre-computed eigenvalues

    Returns:
        ControllabilityInfo with analysis results
    """
    n = A.shape[0]

    # Controllability matrix: [B, AB, A^2B, ...]
    Ctrb = B.copy()
    for i in range(1, n):
        Ctrb = np.hstack([Ctrb, np.linalg.matrix_power(A, i) @ B])

    ctrb_rank = np.linalg.matrix_rank(Ctrb)
    is_fully_controllable = (ctrb_rank == n)

    # SVD of controllability matrix
    s = np.linalg.svd(Ctrb, compute_uv=False)
    min_sv = float(np.min(s))
    max_sv = float(np.max(s))

    # Controllability index
    if max_sv > 0:
        ctrl_index = np.log10(min_sv / max_sv)
    else:
        ctrl_index = -np.inf

    # PBH test for each eigenvalue
    controllable = np.ones(len(eigenvalues), dtype=bool)
    for i, lam in enumerate(eigenvalues):
        pbh = np.hstack([lam * np.eye(n) - A, B])
        if np.linalg.matrix_rank(pbh) < n:
            controllable[i] = False

    # Modal controllability
    _, eigvecs = linalg.eig(A)
    modal_ctrl = np.zeros(n)
    for i in range(n):
        v = eigvecs[:, i]
        modal_ctrl[i] = np.linalg.norm(v.conj() @ B) ** 2
    if np.max(modal_ctrl) > 0:
        modal_ctrl = modal_ctrl / np.max(modal_ctrl)

    return ControllabilityInfo(
        rank=ctrb_rank,
        is_fully_controllable=is_fully_controllable,
        controllability_index=float(ctrl_index),
        min_singular_value=min_sv,
        max_singular_value=max_sv,
        modal_controllability=modal_ctrl,
        controllable_eigenvalues=controllable,
        num_controllable=int(np.sum(controllable)),
    )


def analyze_stability(eigenvalues: np.ndarray) -> StabilityInfo:
    """Analyze system stability from eigenvalues.

    Maps analyze_stability() in LinearSysAnalysis.m.
    """
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    max_real = float(np.max(real_parts))

    if max_real < -1e-10:
        stability_type = "asymptotically_stable"
    elif max_real <= 1e-10:
        stability_type = "marginally_stable"
    else:
        stability_type = "unstable"

    # Damping for oscillatory modes
    complex_mask = np.abs(imag_parts) > 1e-10
    has_oscillatory = bool(np.any(complex_mask))

    max_freq = 0.0
    min_damping = float('inf')

    if has_oscillatory:
        complex_eig = eigenvalues[complex_mask]
        natural_freqs = np.abs(complex_eig)
        damping_ratios = -np.real(complex_eig) / natural_freqs
        max_freq = float(np.max(natural_freqs))
        min_damping = float(np.min(damping_ratios))

    return StabilityInfo(
        type=stability_type,
        margin=-max_real,
        max_real_part=max_real,
        has_oscillatory_modes=has_oscillatory,
        max_frequency=max_freq,
        min_damping_ratio=min_damping,
    )


def analyze_modes(eigenvalues: np.ndarray,
                  eigenvectors: np.ndarray) -> list[ModalInfo]:
    """Analyze individual modes.

    Maps analyze_modes() in LinearSysAnalysis.m.
    """
    modes = []
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]

        if np.abs(np.imag(lam)) < 1e-10:
            # Real mode
            real_val = np.real(lam)
            if real_val < 0:
                tc = -1.0 / real_val
            else:
                tc = float('inf')

            modes.append(ModalInfo(
                type="real",
                eigenvalue=lam,
                time_constant=tc,
                settling_time=4 * tc,
                dominant_states=[int(k) for k in np.where(np.abs(v) > 0.1 * np.max(np.abs(v)))[0]],
            ))
        else:
            # Complex mode
            wn = np.abs(lam)
            zeta = -np.real(lam) / wn
            wd = np.abs(np.imag(lam))

            if np.real(lam) < 0:
                tc = 1.0 / np.abs(np.real(lam))
            else:
                tc = float('inf')

            overshoot = 0.0
            if zeta < 1 and zeta > 0:
                overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100

            modes.append(ModalInfo(
                type="complex",
                eigenvalue=lam,
                natural_frequency=float(wn),
                damping_ratio=float(zeta),
                damped_frequency=float(wd),
                time_constant=tc,
                settling_time=4 * tc,
                overshoot=float(overshoot),
                dominant_states=[int(k) for k in np.where(np.abs(v) > 0.1 * np.max(np.abs(v)))[0]],
            ))

    return modes


def analyze_system(A: np.ndarray, B: np.ndarray) -> SystemAnalysis:
    """Complete system analysis at a trim point.

    Maps LinearSysAnalysis.m.

    Args:
        A: 4x4 system matrix
        B: 4x2 input matrix

    Returns:
        SystemAnalysis with all analysis results
    """
    # Check for NaN
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        return SystemAnalysis()

    eigen_info = analyze_eigenvalues(A)
    ctrl_info = analyze_controllability(A, B, eigen_info.eigenvalues)
    stab_info = analyze_stability(eigen_info.eigenvalues)
    modal_info = analyze_modes(eigen_info.eigenvalues, eigen_info.eigenvectors)

    return SystemAnalysis(
        eigen_info=eigen_info,
        controllability_info=ctrl_info,
        stability_info=stab_info,
        modal_info=modal_info,
    )


def analyze_longitudinal_system(A: np.ndarray, B: np.ndarray,
                                settling_time: float = 0.5,
                                q_tolerance: float = 0.1) -> float:
    """Analyze max q perturbation for given settling time and tolerance.

    Maps analyze_longitudinal_system.m. Estimates the maximum initial
    pitch rate perturbation that can be regulated to |q| < q_tolerance
    within settling_time.

    Args:
        A: 4x4 system matrix
        B: 4x2 input matrix
        settling_time: Desired settling time (s)
        q_tolerance: Acceptable q range (rad/s)

    Returns:
        Maximum initial q perturbation (rad/s)
    """
    eigen_info = analyze_eigenvalues(A)
    modes = analyze_modes(eigen_info.eigenvalues, eigen_info.eigenvectors)

    # q is the 4th state (index 3)
    q_index = 3
    eigvals = eigen_info.eigenvalues
    eigvecs = eigen_info.eigenvectors

    # Find stable modes and compute decay
    total_decay = 0.0
    for i, mode in enumerate(modes):
        if mode.time_constant > 0 and mode.time_constant < float('inf'):
            lam = eigvals[i]
            decay = np.exp(np.real(lam) * settling_time)
            q_participation = np.abs(eigvecs[q_index, i])
            total_decay += q_participation * decay

    if total_decay > 0:
        max_q = q_tolerance / total_decay
    else:
        max_q = 0.0

    return float(max_q)


# ---------------------------------------------------------------------------
# Corridor sweep with linearization
# ---------------------------------------------------------------------------

def _linearize_single_point(i: int, j: int, V: float, theta_deg: float,
                            trim_flag: float, throttle: float,
                            elevator: float, alpha: float,
                            aero: AeroData, xs: float,
                            settling_time: float,
                            q_tolerance: float) -> dict:
    """Linearize and analyze a single (V, theta) point."""
    result = {
        'real_max': 0.0, 'real_min': 0.0,
        'real_num': 0, 'complex_num': 0,
        'ctrl_index': 0.0, 'msv': 0.0,
    }

    if trim_flag != 1:
        return result

    try:
        A, B = linearize_at_point(V, alpha, theta_deg, throttle, elevator,
                                  aero, xs)

        if np.any(np.isnan(A)) or np.any(np.isnan(B)):
            return result

        analysis = analyze_system(A, B)
        max_q = analyze_longitudinal_system(A, B, settling_time, q_tolerance)

        result['real_max'] = analysis.eigen_info.max_real_part
        result['real_min'] = analysis.eigen_info.min_real_part
        result['real_num'] = analysis.eigen_info.real_eigen_count
        result['complex_num'] = analysis.eigen_info.complex_eigen_count
        result['ctrl_index'] = analysis.controllability_info.controllability_index
        result['msv'] = analysis.controllability_info.min_singular_value

    except Exception:
        pass

    return result


def sweep_linearization(mesh: TrimMesh, aero: AeroData,
                        xs: float = 0.0,
                        settling_time: float = 0.5,
                        q_tolerance: float = 0.1,
                        n_jobs: int = 1) -> TrimMesh:
    """Add linearization data to an existing TrimMesh.

    Maps Main.m (linearization) — sweeps over trim grid and computes
    eigenvalue/controllability metrics at each trim point.

    Args:
        mesh: Existing TrimMesh with trim results
        aero: Lookup tables
        xs: Moving mass position
        settling_time: Settling time for q perturbation analysis
        q_tolerance: q tolerance for perturbation analysis
        n_jobs: Number of parallel workers

    Returns:
        TrimMesh with linearization data added
    """
    n_v = len(mesh.trim_V)
    n_theta = len(mesh.trim_theta)

    # Initialize output arrays
    mesh.real_max = np.zeros((n_v, n_theta))
    mesh.real_min = np.zeros((n_v, n_theta))
    mesh.real_num = np.zeros((n_v, n_theta))
    mesh.complex_num = np.zeros((n_v, n_theta))
    mesh.controllability_index = np.zeros((n_v, n_theta))
    mesh.min_singular_value = np.zeros((n_v, n_theta))

    # Build task list
    tasks = []
    for i in range(n_v):
        for j in range(n_theta):
            tasks.append((i, j, mesh.trim_V[i], mesh.trim_theta[j],
                          mesh.trim_flag[i, j], mesh.throttle[i, j],
                          mesh.elevator[i, j], mesh.alpha[i, j]))

    total = len(tasks)

    if n_jobs <= 1:
        for idx, (i, j, V, theta, flag, thr, ele, alpha) in enumerate(tasks):
            res = _linearize_single_point(
                i, j, V, theta, flag, thr, ele, alpha,
                aero, xs, settling_time, q_tolerance)
            _store_lin_result(mesh, i, j, res)
            if flag == 1:
                print(f"[{idx+1}/{total}] Linearized V={V:.1f}, theta={theta:.1f}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, j, V, theta, flag, thr, ele, alpha in tasks:
                future = executor.submit(
                    _linearize_single_point,
                    i, j, V, theta, flag, thr, ele, alpha,
                    aero, xs, settling_time, q_tolerance)
                futures[future] = (i, j, V, theta)

            for idx, future in enumerate(as_completed(futures)):
                i, j, V, theta = futures[future]
                res = future.result()
                _store_lin_result(mesh, i, j, res)
                if mesh.trim_flag[i, j] == 1:
                    print(f"[{idx+1}/{total}] Linearized V={V:.1f}, theta={theta:.1f}")

    return mesh


def _store_lin_result(mesh: TrimMesh, i: int, j: int, result: dict):
    """Store linearization result into mesh arrays."""
    mesh.real_max[i, j] = result['real_max']
    mesh.real_min[i, j] = result['real_min']
    mesh.real_num[i, j] = result['real_num']
    mesh.complex_num[i, j] = result['complex_num']
    mesh.controllability_index[i, j] = result['ctrl_index']
    mesh.min_singular_value[i, j] = result['msv']
