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
from scipy.interpolate import interp1d

from .config import PhysicalConfig


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
        """Build scipy interpolation functions from lookup table data."""
        # Aero coefficients vs alpha (from aero_cfg2.mat)
        alpha = aero_data["alpha"].flatten()
        self._cL_interp = interp1d(alpha, aero_data["cL"].flatten(),
                                   kind="linear", fill_value="extrapolate")
        self._cD_interp = interp1d(alpha, aero_data["cD"].flatten(),
                                   kind="linear", fill_value="extrapolate")
        self._cm_interp = interp1d(alpha, aero_data["cm"].flatten(),
                                   kind="linear", fill_value="extrapolate")

        # Surface increments vs elevator (from aerosurface.mat)
        ele = aero_data["ele"].flatten()
        self._dcL_interp = interp1d(ele, aero_data["dcL"].flatten(),
                                    kind="linear", fill_value="extrapolate")
        self._dcD_interp = interp1d(ele, aero_data["dcD"].flatten(),
                                    kind="linear", fill_value="extrapolate")
        self._dcm_interp = interp1d(ele, aero_data["dcm"].flatten(),
                                    kind="linear", fill_value="extrapolate")

        # Propulsion: throttle -> thrust (from prop.mat)
        throttle = aero_data["throttle"].flatten()
        thrust = aero_data["thrust"].flatten()
        self._thrust_interp = interp1d(throttle, thrust,
                                       kind="linear", fill_value="extrapolate")

    def _compute_aero(self, V: float, alpha_deg: float, ele_deg: float) -> tuple[float, float, float]:
        """Compute aerodynamic forces and moment.

        Returns (L, D, my_aero) in body frame.
        """
        Q = 0.5 * self.phys.rho * self.phys.S * V * V

        # Lookup coefficients
        cL = float(self._cL_interp(alpha_deg))
        cD = float(self._cD_interp(alpha_deg))
        cm = float(self._cm_interp(alpha_deg))

        # Surface increments
        dcL = float(self._dcL_interp(ele_deg))
        dcD = float(self._dcD_interp(ele_deg))
        dcm = float(self._dcm_interp(ele_deg))

        CL = cL + dcL
        CD = cD + dcD
        Cm = cm + dcm

        L = Q * CL
        D = Q * CD
        my_aero = Q * self.phys.c * Cm

        return L, D, my_aero

    def _compute_prop(self, throttle: float) -> float:
        """Compute propeller thrust from throttle command."""
        thrust_raw = float(self._thrust_interp(throttle))
        return thrust_raw

    def _ode_rhs(self, x: np.ndarray, F: np.ndarray, my: float,
                 state_s: np.ndarray) -> np.ndarray:
        """Compute state derivatives.

        From Stateflow chart_33:
        x = [u, w, theta, q, h]
        F = [Fx, Fz]
        state_s = [xs, vs, vsdot] (moving mass)
        """
        u, w, theta, q, h = x
        fx, fz = F
        xs, vs, vsdot = state_s

        p = self.phys

        qdot = (my - p.ms * xs * p.g * np.cos(theta) - 2 * p.ms * xs * vs * q) / \
               (p.Jy + p.ms * xs * xs)
        udot = fx / p.m - p.mu * vsdot
        wdot = fz / p.m - p.mu * (-qdot * xs - q * vs)
        thetadot = q
        hdot = w * np.cos(theta) - u * np.sin(theta)

        return np.array([udot, wdot, thetadot, qdot, hdot])

    def _compute_forces(self, alpha: float, theta: float, L: float, D: float,
                        my_aero: float, Fprop: float) -> tuple[np.ndarray, float]:
        """Compute total force vector and moment.

        From Stateflow chart_18:
        F = rotation_matrix @ [-L; -D] + [2*Fprop; 0] + m*g*[-sin(theta); cos(theta)]
        my = my_aero
        """
        p = self.phys
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        st = np.sin(theta)
        ct = np.cos(theta)

        # Aero forces in body frame (wind-to-body rotation)
        Fx_aero = -sa * (-L) + ca * (-D)
        Fz_aero = ca * (-L) + sa * (-D)

        # Propulsion (factor of 2 from Simulink model)
        Fx_prop = 2 * Fprop

        # Gravity in body frame
        Fx_grav = p.m * p.g * (-st)
        Fz_grav = p.m * p.g * ct

        Fx = Fx_aero + Fx_prop + Fx_grav
        Fz = Fz_aero + Fz_grav

        return np.array([Fx, Fz]), my_aero

    def step(self, state: np.ndarray, action_phys: np.ndarray, dt: float) -> np.ndarray:
        """Advance state by one timestep using RK4 integration.

        Args:
            state: 7D [u, w, V, alpha, theta, q, h]
            action_phys: 2D [throttle, elevator_deg] (physical units)
            dt: timestep (0.02s)

        Returns:
            new_state: 7D [u, w, V, alpha, theta, q, h]
        """
        throttle = np.clip(action_phys[0], 0.0, 1.0)
        ele_deg = action_phys[1]

        # Extract integrated states [u, w, theta, q, h]
        x = state[[0, 1, 4, 5, 6]].copy()

        # Moving mass (simplified: us=0 for longitudinal, state_s = [0, 0, 0])
        state_s = np.array([0.0, 0.0, 0.0])

        def compute_xdot(x_cur):
            alpha_deg = np.degrees(np.arctan2(x_cur[1], x_cur[0])) if (x_cur[0]**2 + x_cur[1]**2) > 1e-10 else 90.0
            V = np.sqrt(x_cur[0]**2 + x_cur[1]**2)
            L, D, my_aero = self._compute_aero(V, alpha_deg, ele_deg)
            Fprop = self._compute_prop(throttle)
            F, my = self._compute_forces(np.radians(alpha_deg), x_cur[2], L, D, my_aero, Fprop)
            return self._ode_rhs(x_cur, F, my, state_s)

        # RK4 integration
        k1 = compute_xdot(x)
        k2 = compute_xdot(x + 0.5 * dt * k1)
        k3 = compute_xdot(x + 0.5 * dt * k2)
        k4 = compute_xdot(x + dt * k3)

        x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Compute derived quantities
        V_new = np.sqrt(x_new[0]**2 + x_new[1]**2)
        if V_new > 1e-10:
            alpha_new = np.arctan2(x_new[1], x_new[0])
        else:
            alpha_new = np.pi / 2

        return np.array([x_new[0], x_new[1], V_new, alpha_new,
                         x_new[2], x_new[3], x_new[4]])
