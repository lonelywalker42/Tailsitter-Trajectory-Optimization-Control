"""Tailsitter longitudinal dynamics model.

Reverse-engineered from mdl/Lon_mdl.slx:
- Stateflow chart_18: force equation (aero + propulsion + gravity)
- Stateflow chart_33: 5-DOF ODE with moving mass coupling
- Stateflow chart_48: V = sqrt(u^2 + w^2), alpha = asin(w/V)
- Stateflow chart_57: induced velocity (momentum theory)
- system_7/20: aerodynamic lookup tables (Q, L, D, Cm)
- system_88: propulsion lookup (throttle -> thrust)
- system_80: moving mass dynamics (Transfer Fcn + integrators)
"""

import numpy as np
import numba as nb

from .config import PhysicalConfig


# ---------------------------------------------------------------------------
# Numba JIT kernel — all dynamics in a single compiled function
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _lerp(x, xp, fp):
    """Linear interpolation (Numba-compatible replacement for scipy.interp1d).

    np.interp clamps at boundaries (no extrapolation). This matches
    interp1d(fill_value="extrapolate") for values within the data range,
    which is the normal operating envelope.
    """
    return np.interp(x, xp, fp)


@nb.njit(cache=True)
def _compute_xdot(x_cur, throttle, ele_deg,
                   rho, S, c, m, g, Jy, ms, mu,
                   alpha_arr, cL_arr, cD_arr, cm_arr,
                   ele_arr, dcL_arr, dcD_arr, dcm_arr,
                   thr_arr, thrust_arr):
    """Compute ODE right-hand side for a single state vector.

    Args:
        x_cur: 5D [u, w, theta, q, h]
        throttle: throttle command [0, 1]
        ele_deg: elevator deflection [deg]
        Physical params: rho, S, c, m, g, Jy, ms, mu
        Lookup arrays: alpha_arr/cL_arr/cD_arr/cm_arr (aero vs alpha),
                       ele_arr/dcL_arr/dcD_arr/dcm_arr (increments vs elevator),
                       thr_arr/thrust_arr (propulsion)

    Returns:
        5D state derivative [udot, wdot, thetadot, qdot, hdot]
    """
    u = x_cur[0]
    w = x_cur[1]
    theta = x_cur[2]
    q = x_cur[3]
    h = x_cur[4]

    # Angle of attack and airspeed
    V_sq = u * u + w * w
    if V_sq > 1e-10:
        alpha_deg = np.degrees(np.arctan2(w, u))
        V = np.sqrt(V_sq)
    else:
        alpha_deg = 90.0
        V = 0.0

    # Aero coefficients via interpolation
    Q = 0.5 * rho * S * V * V
    cL = _lerp(alpha_deg, alpha_arr, cL_arr)
    cD = _lerp(alpha_deg, alpha_arr, cD_arr)
    cm = _lerp(alpha_deg, alpha_arr, cm_arr)

    dcL = _lerp(ele_deg, ele_arr, dcL_arr)
    dcD = _lerp(ele_deg, ele_arr, dcD_arr)
    dcm = _lerp(ele_deg, ele_arr, dcm_arr)

    L = Q * (cL + dcL)
    D = Q * (cD + dcD)
    my_aero = Q * c * (cm + dcm)

    # Propulsion
    Fprop = _lerp(throttle, thr_arr, thrust_arr)

    # Forces in body frame (wind-to-body rotation + gravity)
    alpha_rad = np.radians(alpha_deg)
    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)
    st = np.sin(theta)
    ct = np.cos(theta)

    Fx_aero = -sa * (-L) + ca * (-D)
    Fz_aero = ca * (-L) + sa * (-D)

    Fx = Fx_aero + 2.0 * Fprop + m * g * (-st)
    Fz = Fz_aero + m * g * ct

    # State derivatives (state_s = [0,0,0] for longitudinal)
    udot = Fx / m
    wdot = Fz / m
    thetadot = q
    qdot = my_aero / Jy
    hdot = w * ct - u * st

    return np.array([udot, wdot, thetadot, qdot, hdot])


@nb.njit(cache=True)
def _rk4_step_jit(x, throttle, ele_deg, dt,
                   rho, S, c, m, g, Jy, ms, mu,
                   alpha_arr, cL_arr, cD_arr, cm_arr,
                   ele_arr, dcL_arr, dcD_arr, dcm_arr,
                   thr_arr, thrust_arr):
    """Single RK4 integration step (Numba JIT compiled).

    Args:
        x: 5D state [u, w, theta, q, h]
        throttle, ele_deg: control inputs
        dt: timestep
        Physical params and lookup arrays (see _compute_xdot)

    Returns:
        5D new state [u_new, w_new, theta_new, q_new, h_new]
    """
    k1 = _compute_xdot(x, throttle, ele_deg,
                        rho, S, c, m, g, Jy, ms, mu,
                        alpha_arr, cL_arr, cD_arr, cm_arr,
                        ele_arr, dcL_arr, dcD_arr, dcm_arr,
                        thr_arr, thrust_arr)

    x2 = x + 0.5 * dt * k1
    k2 = _compute_xdot(x2, throttle, ele_deg,
                        rho, S, c, m, g, Jy, ms, mu,
                        alpha_arr, cL_arr, cD_arr, cm_arr,
                        ele_arr, dcL_arr, dcD_arr, dcm_arr,
                        thr_arr, thrust_arr)

    x3 = x + 0.5 * dt * k2
    k3 = _compute_xdot(x3, throttle, ele_deg,
                        rho, S, c, m, g, Jy, ms, mu,
                        alpha_arr, cL_arr, cD_arr, cm_arr,
                        ele_arr, dcL_arr, dcD_arr, dcm_arr,
                        thr_arr, thrust_arr)

    x4 = x + dt * k3
    k4 = _compute_xdot(x4, throttle, ele_deg,
                        rho, S, c, m, g, Jy, ms, mu,
                        alpha_arr, cL_arr, cD_arr, cm_arr,
                        ele_arr, dcL_arr, dcD_arr, dcm_arr,
                        thr_arr, thrust_arr)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Python wrapper class
# ---------------------------------------------------------------------------

class LongitudinalDynamics:
    """Tailsitter longitudinal dynamics model.

    Replaces the Simulink plant model Lon_mdl.slx.
    State vector: [u, w, V, alpha, theta, q, h]
    """

    def __init__(self, phys: PhysicalConfig, aero_data: dict):
        """
        Args:
            phys: Physical parameters
            aero_data: Dict loaded from processed .npy files
                       Expected keys: alpha, cL, cD, cm (from aero_cfg2),
                       ele, dcL, dcD, dcm (from aerosurface),
                       throttle, thrust (from prop)
        """
        self.phys = phys
        self._build_interpolators(aero_data)

    def _build_interpolators(self, aero_data: dict):
        """Store raw lookup table arrays for Numba JIT interpolation."""
        # Aero coefficients vs alpha (from aero_cfg2.mat)
        self._alpha_arr = aero_data["alpha"].flatten().astype(np.float64)
        self._cL_arr = aero_data["cL"].flatten().astype(np.float64)
        self._cD_arr = aero_data["cD"].flatten().astype(np.float64)
        self._cm_arr = aero_data["cm"].flatten().astype(np.float64)

        # Surface increments vs elevator (from aerosurface.mat)
        self._ele_arr = aero_data["ele"].flatten().astype(np.float64)
        self._dcL_arr = aero_data["dcL"].flatten().astype(np.float64)
        self._dcD_arr = aero_data["dcD"].flatten().astype(np.float64)
        self._dcm_arr = aero_data["dcm"].flatten().astype(np.float64)

        # Propulsion: throttle -> thrust (from prop.mat)
        self._thr_arr = aero_data["throttle"].flatten().astype(np.float64)
        self._thrust_arr = aero_data["thrust"].flatten().astype(np.float64)

    def step(self, state: np.ndarray, action_phys: np.ndarray, dt: float) -> np.ndarray:
        """Advance state by one timestep using RK4 integration.

        Args:
            state: 7D [u, w, V, alpha, theta, q, h]
            action_phys: 2D [throttle, elevator_deg] (physical units)
            dt: timestep (0.02s)

        Returns:
            new_state: 7D [u, w, V, alpha, theta, q, h]
        """
        throttle = float(np.clip(action_phys[0], 0.0, 1.0))
        ele_deg = float(action_phys[1])

        # Extract integrated states [u, w, theta, q, h]
        x = np.ascontiguousarray(state[[0, 1, 4, 5, 6]], dtype=np.float64)

        p = self.phys

        # Call Numba JIT kernel
        x_new = _rk4_step_jit(
            x, throttle, ele_deg, dt,
            p.rho, p.S, p.c, p.m, p.g, p.Jy, p.ms, p.mu,
            self._alpha_arr, self._cL_arr, self._cD_arr, self._cm_arr,
            self._ele_arr, self._dcL_arr, self._dcD_arr, self._dcm_arr,
            self._thr_arr, self._thrust_arr,
        )

        # Compute derived quantities
        V_new = np.sqrt(x_new[0] ** 2 + x_new[1] ** 2)
        if V_new > 1e-10:
            alpha_new = np.arctan2(x_new[1], x_new[0])
        else:
            alpha_new = np.pi / 2

        return np.array([x_new[0], x_new[1], V_new, alpha_new,
                         x_new[2], x_new[3], x_new[4]])
