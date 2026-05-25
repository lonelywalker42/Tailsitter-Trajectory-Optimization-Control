"""Reward function for the tailsitter longitudinal control environment.

Reverse-engineered from RL/LonEnv.slx:
- system_857.xml: reward computation (5 components)
- system_865/874/883/892.xml: nonlinear shaping functions
- system_226.xml: success detection integrator
"""

import numpy as np

from .config import EnvConfig


class RewardCalculator:
    """Computes reward matching the Simulink LonEnv Reward subsystem."""

    def __init__(self, config: EnvConfig):
        self.Ts = config.Ts

    @staticmethod
    def shaping_func(x: float) -> float:
        """Nonlinear shaping: 1 / sqrt(0.1 * x^2 + 1). Peaks at 1.0 for x=0."""
        return 1.0 / np.sqrt(0.1 * x * x + 1.0)

    def compute(
        self,
        state: np.ndarray,
        target: np.ndarray,
        t: float,
        success_integral: float,
    ) -> tuple[float, float]:
        """Compute reward and updated success integral.

        Args:
            state: 7D [u, w, V, alpha, theta, q, h]
            target: 3D [V_target, alpha_target, theta_target]
            t: elapsed time [s]
            success_integral: running integral of success indicator

        Returns:
            (reward, updated_success_integral)
        """
        V = state[2]
        alpha = state[3]
        theta = state[4]
        Vt, alpha_t, theta_t = target

        # State error for success detection (weighted, mixed units)
        err_V = 0.5 * (V - Vt)
        err_alpha = 0.5 * 57.3 * (alpha - alpha_t)  # rad -> deg
        err_theta = 0.5 * 57.3 * (theta - theta_t)   # rad -> deg
        state_error_norm = np.sqrt(err_V**2 + err_alpha**2 + err_theta**2)

        # Success integral (binary: 1 if norm < 2, else 0)
        success_indicator = 1.0 if state_error_norm < 2.0 else 0.0
        success_integral = min(success_integral + success_indicator * self.Ts, 1.0)

        # Reward components
        r_success = 1000.0 * success_integral
        r_V = self.shaping_func(5.0 * abs(V - Vt))
        r_theta = self.shaping_func(abs(np.degrees(theta - theta_t)))
        r_alpha = self.shaping_func(abs(np.degrees(alpha - alpha_t)))
        r_dh = 0.05 * (state[1] ** 2)  # w (body-z velocity ≈ dh/dt)

        time_coef = -0.05 * np.exp(0.1 * t)

        reward = time_coef + r_success + r_V + r_theta + r_alpha + r_dh
        return reward, success_integral
