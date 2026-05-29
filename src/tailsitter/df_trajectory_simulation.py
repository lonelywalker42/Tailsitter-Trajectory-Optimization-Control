"""Trajectory tracking simulation for DF-optimized trajectories.

Replaces Simulink-dependent scripts (sim_main.m, sim_main_baseline.m,
sim_compare.m, transition_sim_compare.m) with Python-based open-loop
simulation using dynamics.py + error analysis.

The simulation feeds DF reference controls (throttle, elevator) into the
nonlinear dynamics model and records the actual state evolution, then
computes RMSE and end-point tracking errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import LongitudinalDynamics
from .df_trajectory_optimization import DFTrajectoryResult
from .baseline_trajectory import BaselineTrajectoryResult
from .trim import G, R2D


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of open-loop trajectory tracking simulation."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))

    # Actual (simulated) states — all in deg/deg/s for consistency with MATLAB
    actual_V: np.ndarray = field(default_factory=lambda: np.array([]))       # m/s
    actual_gamma: np.ndarray = field(default_factory=lambda: np.array([]))   # deg
    actual_alpha: np.ndarray = field(default_factory=lambda: np.array([]))   # deg
    actual_theta: np.ndarray = field(default_factory=lambda: np.array([]))   # deg
    actual_q: np.ndarray = field(default_factory=lambda: np.array([]))       # deg/s
    actual_h: np.ndarray = field(default_factory=lambda: np.array([]))       # m
    actual_x: np.ndarray = field(default_factory=lambda: np.array([]))       # m

    # Reference states (interpolated to uniform time grid)
    ref_V: np.ndarray = field(default_factory=lambda: np.array([]))          # m/s
    ref_gamma: np.ndarray = field(default_factory=lambda: np.array([]))      # deg
    ref_alpha: np.ndarray = field(default_factory=lambda: np.array([]))      # deg
    ref_theta: np.ndarray = field(default_factory=lambda: np.array([]))      # deg
    ref_q: np.ndarray = field(default_factory=lambda: np.array([]))          # deg/s
    ref_h: np.ndarray = field(default_factory=lambda: np.array([]))          # m

    # Control inputs applied
    throttle: np.ndarray = field(default_factory=lambda: np.array([]))       # [0, 1]
    elevator: np.ndarray = field(default_factory=lambda: np.array([]))       # deg


@dataclass
class ErrorAnalysis:
    """Tracking error analysis results (matches MATLAB analyze_simulation_error)."""
    # RMSE for each channel
    rmse_V: float = 0.0       # m/s
    rmse_gamma: float = 0.0   # deg
    rmse_theta: float = 0.0   # deg
    rmse_q: float = 0.0       # deg/s
    rmse_alpha: float = 0.0   # deg
    rmse_height: float = 0.0  # m

    # Maximum absolute deviation
    max_V: float = 0.0
    max_gamma: float = 0.0
    max_theta: float = 0.0
    max_q: float = 0.0
    max_alpha: float = 0.0
    max_height: float = 0.0

    # End-point error (actual - reference at final time)
    end_V: float = 0.0
    end_gamma: float = 0.0
    end_theta: float = 0.0
    end_q: float = 0.0
    end_alpha: float = 0.0
    end_height: float = 0.0

    # Performance grades
    grade_V: str = ""
    grade_gamma: str = ""
    grade_theta: str = ""
    grade_q: str = ""


# ---------------------------------------------------------------------------
# Performance grading thresholds (matching MATLAB)
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS = {
    "V":     [(0.5, "excellent"), (1.0, "good"), (2.0, "fair")],
    "gamma": [(0.5, "excellent"), (1.0, "good"), (2.0, "fair")],
    "theta": [(1.0, "excellent"), (2.0, "good"), (3.0, "fair")],
    "q":     [(2.0, "excellent"), (5.0, "good"), (10.0, "fair")],
}


def _grade(rmse: float, channel: str) -> str:
    """Assign performance grade based on RMSE threshold."""
    for threshold, label in _GRADE_THRESHOLDS[channel]:
        if rmse < threshold:
            return label
    return "poor"


# ---------------------------------------------------------------------------
# Thrust-to-throttle conversion
# ---------------------------------------------------------------------------

def _thrust_to_throttle(thrust_N: np.ndarray, aero_data: dict) -> np.ndarray:
    """Convert thrust (N) to throttle [0, 1] via inverse lookup.

    Forward: Fprop = G * np.interp(throttle, aero.throttle, aero.thrust)
    Inverse: throttle = np.interp(thrust_N / G, aero.thrust, aero.throttle)
    """
    thr_arr = aero_data["throttle"].flatten()
    thrust_arr = aero_data["thrust"].flatten()
    return np.interp(thrust_N / G, thrust_arr, thr_arr)


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------

def simulate_df_trajectory(
    traj: DFTrajectoryResult,
    dyn: LongitudinalDynamics,
    aero_data: dict,
    dt: float = 0.02,
    h0: float = 0.0,
) -> SimulationResult:
    """Simulate aircraft following a DF-optimized trajectory.

    Open-loop: feeds reference throttle and elevator directly into the
    dynamics model at each timestep.

    Args:
        traj: DF trajectory optimization result
        dyn: Dynamics model instance
        aero_data: Aero data dict (for thrust-to-throttle conversion)
        dt: Simulation timestep [s]
        h0: Initial altitude [m]

    Returns:
        SimulationResult with actual and reference states
    """
    # Uniform time grid
    t_end = float(traj.time[-1])
    t_uniform = np.arange(0, t_end, dt)

    # Resample reference to uniform grid
    ref_V = np.interp(t_uniform, traj.time, traj.V)
    ref_gamma = np.degrees(np.interp(t_uniform, traj.time, traj.gamma))
    ref_alpha = np.degrees(np.interp(t_uniform, traj.time, traj.alpha))
    ref_theta = np.degrees(np.interp(t_uniform, traj.time, traj.theta))
    ref_q = np.degrees(np.interp(t_uniform, traj.time, traj.q))

    # Compute reference height by integrating V*sin(gamma)
    gamma_rad = np.radians(ref_gamma)
    ref_h = np.zeros_like(t_uniform)
    for i in range(1, len(t_uniform)):
        ref_h[i] = ref_h[i - 1] + dt * ref_V[i] * np.sin(gamma_rad[i])

    # Convert controls
    throttle_ref = _thrust_to_throttle(np.interp(t_uniform, traj.time, traj.throttle), aero_data)
    elevator_ref = np.interp(t_uniform, traj.time, traj.elevator)

    # Initial state from reference
    V0 = ref_V[0]
    alpha0_rad = traj.alpha[0]
    theta0_rad = traj.theta[0]
    q0 = traj.q[0]
    u0 = V0 * np.cos(alpha0_rad)
    w0 = V0 * np.sin(alpha0_rad)
    state = np.array([u0, w0, V0, alpha0_rad, theta0_rad, q0, h0])

    # Simulation loop
    n_steps = len(t_uniform)
    actual_V = np.zeros(n_steps)
    actual_gamma = np.zeros(n_steps)
    actual_alpha = np.zeros(n_steps)
    actual_theta = np.zeros(n_steps)
    actual_q = np.zeros(n_steps)
    actual_h = np.zeros(n_steps)
    actual_x = np.zeros(n_steps)

    for i in range(n_steps):
        # Record current state
        u, w, V, alpha, theta, q, h = state
        actual_V[i] = V
        actual_alpha[i] = np.degrees(alpha)
        actual_theta[i] = np.degrees(theta)
        actual_q[i] = np.degrees(q)
        actual_h[i] = h

        # Compute gamma = theta - alpha
        actual_gamma[i] = np.degrees(theta - alpha)

        # Approximate longitudinal position: x += V * cos(gamma) * dt
        gamma_actual = theta - alpha
        if i > 0:
            actual_x[i] = actual_x[i - 1] + dt * V * np.cos(gamma_actual)

        # Apply control
        action = np.array([float(throttle_ref[i]), float(elevator_ref[i])])

        # Step dynamics (except at last point)
        if i < n_steps - 1:
            state = dyn.step(state, action, dt)

    return SimulationResult(
        time=t_uniform,
        actual_V=actual_V,
        actual_gamma=actual_gamma,
        actual_alpha=actual_alpha,
        actual_theta=actual_theta,
        actual_q=actual_q,
        actual_h=actual_h,
        actual_x=actual_x,
        ref_V=ref_V,
        ref_gamma=ref_gamma,
        ref_alpha=ref_alpha,
        ref_theta=ref_theta,
        ref_q=ref_q,
        ref_h=ref_h,
        throttle=throttle_ref,
        elevator=elevator_ref,
    )


def simulate_baseline_trajectory(
    result: BaselineTrajectoryResult,
    dyn: LongitudinalDynamics,
    aero_data: dict,
    dt: float = 0.02,
    acc: float = 5.0,
    h0: float = 0.0,
) -> SimulationResult:
    """Simulate aircraft following a baseline corridor path.

    The baseline path has non-uniform spacing in V; time is computed from
    constant acceleration assumption: dt_i = (V_i - V_{i-1}) / acc.

    Args:
        result: Baseline trajectory result
        dyn: Dynamics model instance
        aero_data: Aero data dict (for thrust-to-throttle conversion)
        dt: Simulation timestep [s]
        acc: Assumed acceleration for time computation [m/s^2]
        h0: Initial altitude [m]

    Returns:
        SimulationResult with actual and reference states
    """
    path = result.optimal_path  # N×7: [V, γ°, θ°, α°, T(N), ele°, qmax°]

    # Compute non-uniform time from V and acceleration
    V_path = path[:, 0]
    dV = np.diff(V_path)
    dt_segments = np.abs(dV) / acc
    time_path = np.concatenate([[0], np.cumsum(dt_segments)])

    # Compute pitch rate from theta (numerical derivative)
    theta_deg = path[:, 2]
    q_path = np.gradient(theta_deg, time_path)  # deg/s

    # Extract reference data
    gamma_path = path[:, 1]   # deg
    alpha_path = path[:, 3]   # deg
    T_path = path[:, 4]       # thrust N
    ele_path = path[:, 5]     # deg

    # Uniform time grid
    t_end = time_path[-1]
    t_uniform = np.arange(0, t_end, dt)

    # Resample to uniform grid
    ref_V = np.interp(t_uniform, time_path, V_path)
    ref_gamma = np.interp(t_uniform, time_path, gamma_path)
    ref_alpha = np.interp(t_uniform, time_path, alpha_path)
    ref_theta = np.interp(t_uniform, time_path, theta_deg)
    ref_q = np.interp(t_uniform, time_path, q_path)

    # Compute reference height
    gamma_rad = np.radians(ref_gamma)
    ref_h = np.zeros_like(t_uniform)
    for i in range(1, len(t_uniform)):
        ref_h[i] = ref_h[i - 1] + dt * ref_V[i] * np.sin(gamma_rad[i])

    # Convert controls
    throttle_ref = _thrust_to_throttle(np.interp(t_uniform, time_path, T_path), aero_data)
    elevator_ref = np.interp(t_uniform, time_path, ele_path)

    # Initial state
    V0 = ref_V[0]
    alpha0_rad = np.radians(ref_alpha[0])
    theta0_rad = np.radians(ref_theta[0])
    q0_rad = np.radians(ref_q[0])
    u0 = V0 * np.cos(alpha0_rad)
    w0 = V0 * np.sin(alpha0_rad)
    state = np.array([u0, w0, V0, alpha0_rad, theta0_rad, q0_rad, h0])

    # Simulation loop
    n_steps = len(t_uniform)
    actual_V = np.zeros(n_steps)
    actual_gamma = np.zeros(n_steps)
    actual_alpha = np.zeros(n_steps)
    actual_theta = np.zeros(n_steps)
    actual_q = np.zeros(n_steps)
    actual_h = np.zeros(n_steps)
    actual_x = np.zeros(n_steps)

    for i in range(n_steps):
        u, w, V, alpha, theta, q, h = state
        actual_V[i] = V
        actual_alpha[i] = np.degrees(alpha)
        actual_theta[i] = np.degrees(theta)
        actual_q[i] = np.degrees(q)
        actual_h[i] = h
        actual_gamma[i] = np.degrees(theta - alpha)

        gamma_actual = theta - alpha
        if i > 0:
            actual_x[i] = actual_x[i - 1] + dt * V * np.cos(gamma_actual)

        action = np.array([float(throttle_ref[i]), float(elevator_ref[i])])

        if i < n_steps - 1:
            state = dyn.step(state, action, dt)

    return SimulationResult(
        time=t_uniform,
        actual_V=actual_V,
        actual_gamma=actual_gamma,
        actual_alpha=actual_alpha,
        actual_theta=actual_theta,
        actual_q=actual_q,
        actual_h=actual_h,
        actual_x=actual_x,
        ref_V=ref_V,
        ref_gamma=ref_gamma,
        ref_alpha=ref_alpha,
        ref_theta=ref_theta,
        ref_q=ref_q,
        ref_h=ref_h,
        throttle=throttle_ref,
        elevator=elevator_ref,
    )


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_tracking_error(sim: SimulationResult) -> ErrorAnalysis:
    """Compute tracking error metrics from simulation result.

    Matches MATLAB transition_sim_compare.m :: analyze_simulation_error().

    Args:
        sim: Simulation result with actual and reference states

    Returns:
        ErrorAnalysis with RMSE, max deviation, end-point error, and grades
    """
    # Compute errors
    err_V = sim.actual_V - sim.ref_V
    err_gamma = sim.actual_gamma - sim.ref_gamma
    err_theta = sim.actual_theta - sim.ref_theta
    err_q = sim.actual_q - sim.ref_q
    err_alpha = sim.actual_alpha - sim.ref_alpha
    err_h = sim.actual_h - sim.ref_h

    rmse_V = float(np.sqrt(np.mean(err_V ** 2)))
    rmse_gamma = float(np.sqrt(np.mean(err_gamma ** 2)))
    rmse_theta = float(np.sqrt(np.mean(err_theta ** 2)))
    rmse_q = float(np.sqrt(np.mean(err_q ** 2)))
    rmse_alpha = float(np.sqrt(np.mean(err_alpha ** 2)))
    rmse_height = float(np.sqrt(np.mean(err_h ** 2)))

    return ErrorAnalysis(
        rmse_V=rmse_V,
        rmse_gamma=rmse_gamma,
        rmse_theta=rmse_theta,
        rmse_q=rmse_q,
        rmse_alpha=rmse_alpha,
        rmse_height=rmse_height,
        max_V=float(np.max(np.abs(err_V))),
        max_gamma=float(np.max(np.abs(err_gamma))),
        max_theta=float(np.max(np.abs(err_theta))),
        max_q=float(np.max(np.abs(err_q))),
        max_alpha=float(np.max(np.abs(err_alpha))),
        max_height=float(np.max(np.abs(err_h))),
        end_V=float(err_V[-1]),
        end_gamma=float(err_gamma[-1]),
        end_theta=float(err_theta[-1]),
        end_q=float(err_q[-1]),
        end_alpha=float(err_alpha[-1]),
        end_height=float(err_h[-1]),
        grade_V=_grade(rmse_V, "V"),
        grade_gamma=_grade(rmse_gamma, "gamma"),
        grade_theta=_grade(rmse_theta, "theta"),
        grade_q=_grade(rmse_q, "q"),
    )


def print_error_report(error: ErrorAnalysis, label: str = "") -> None:
    """Print formatted error analysis report to stdout.

    Args:
        error: Error analysis result
        label: Optional label (e.g., "Proposed" or "Baseline")
    """
    prefix = f"[{label}] " if label else ""

    print(f"\n{'=' * 60}")
    print(f"{prefix}Tracking Error Analysis")
    print(f"{'=' * 60}")

    print(f"\n{'Parameter':<25} {'RMSE':>12} {'Max Dev':>12} {'End Err':>12}")
    print("-" * 61)
    print(f"{'Velocity [m/s]':<25} {error.rmse_V:>12.4f} {error.max_V:>12.4f} {error.end_V:>12.4f}")
    print(f"{'FPA [deg]':<25} {error.rmse_gamma:>12.4f} {error.max_gamma:>12.4f} {error.end_gamma:>12.4f}")
    print(f"{'Pitch [deg]':<25} {error.rmse_theta:>12.4f} {error.max_theta:>12.4f} {error.end_theta:>12.4f}")
    print(f"{'Pitch Rate [deg/s]':<25} {error.rmse_q:>12.4f} {error.max_q:>12.4f} {error.end_q:>12.4f}")
    print(f"{'AoA [deg]':<25} {error.rmse_alpha:>12.4f} {error.max_alpha:>12.4f} {error.end_alpha:>12.4f}")
    print(f"{'Altitude [m]':<25} {error.rmse_height:>12.4f} {error.max_height:>12.4f} {error.end_height:>12.4f}")

    print(f"\n{'Performance Grade':<25} {'Level':>12}")
    print("-" * 37)
    print(f"{'Velocity':<25} {error.grade_V:>12}")
    print(f"{'FPA':<25} {error.grade_gamma:>12}")
    print(f"{'Pitch':<25} {error.grade_theta:>12}")
    print(f"{'Pitch Rate':<25} {error.grade_q:>12}")
    print()
