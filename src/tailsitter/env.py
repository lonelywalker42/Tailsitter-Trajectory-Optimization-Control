"""Gymnasium environment for tailsitter longitudinal control.

Replaces the Simulink LonEnv.slx RL environment.
Observation space (11D): [V, alpha, theta, q, h, dt_last, de_last, dV, dalpha, dtheta, cmd]
Action space (2D): [dt, de] in [-1, 1]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import EnvConfig, PhysicalConfig
from .dynamics import LongitudinalDynamics
from .normalization import Normalizer
from .reset import EpisodeResetter
from .reward import RewardCalculator


class TailsitterLongitudinalEnv(gym.Env):
    """Gymnasium environment for tailsitter longitudinal control.

    Replaces the Simulink LonEnv.slx RL environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dynamics: LongitudinalDynamics,
        reward_calc: RewardCalculator,
        normalizer: Normalizer,
        resetter: EpisodeResetter,
        env_config: EnvConfig,
        phys_config: PhysicalConfig,
        reset_mode: str = "sac",
    ):
        super().__init__()
        self.dynamics = dynamics
        self.reward_calc = reward_calc
        self.normalizer = normalizer
        self.resetter = resetter
        self.cfg = env_config
        self.phys = phys_config
        self.reset_mode = reset_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._state = None
        self._target = None
        self._cmd = None
        self._t = 0.0
        self._step_count = 0
        self._dt_last = 0.0
        self._de_last = 0.0
        self._success_integral = 0.0
        self._max_steps = int(self.cfg.Tf / self.cfg.Ts)

        # Actuator filter state
        self._dt_filtered = 0.0
        self._de_filtered = 0.0

    def _build_observation(self) -> np.ndarray:
        """Construct 11D observation from current state and history."""
        V = self._state[2]
        alpha = self._state[3]
        theta = self._state[4]
        q = self._state[5]
        h = self._state[6]

        dV = self._target[0] - V
        dalpha = self._target[1] - alpha
        dtheta = self._target[2] - theta

        raw_obs = np.array([
            V, alpha, theta, q, h,
            self._dt_last, self._de_last,
            dV, dalpha, dtheta,
            float(self._cmd),
        ], dtype=np.float64)

        return self.normalizer.normalize_obs(raw_obs).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.reset_mode == "sac":
            init_state, target, cmd = self.resetter.reset_sac()
        else:
            init_state, target, cmd = self.resetter.reset_ppo()

        self._state = init_state.copy()
        self._target = target.copy()
        self._cmd = cmd
        self._t = 0.0
        self._step_count = 0
        self._dt_last = 0.0
        self._de_last = 0.0
        self._success_integral = 0.0
        self._dt_filtered = 0.0
        self._de_filtered = 0.0

        obs = self._build_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Denormalize action to physical units
        action_phys = self.normalizer.denormalize_action(action)

        # Apply actuator dynamics (first-order lag: tau=0.001s)
        # Discrete: u_new = u_prev + (Ts/(tau+Ts)) * (u_cmd - u_prev)
        tau = self.phys.actuator_tau
        Ts = self.cfg.Ts
        alpha_act = Ts / (tau + Ts)

        dt_cmd = action_phys[0]
        de_cmd = action_phys[1]

        self._dt_filtered += alpha_act * (dt_cmd - self._dt_filtered)
        self._de_filtered += alpha_act * (de_cmd - self._de_filtered)

        # Elevator sign inversion (Simulink Gain = -1)
        dt_phys = self._dt_filtered
        de_phys = self.phys.elevator_sign * self._de_filtered

        # Advance dynamics
        self._state = self.dynamics.step(
            self._state, np.array([dt_phys, de_phys]), self.cfg.Ts
        )

        # Update history (store normalized action, matching Simulink Delay block)
        self._dt_last = action[0]
        self._de_last = action[1]
        self._t += self.cfg.Ts
        self._step_count += 1

        # Compute reward
        reward, self._success_integral = self.reward_calc.compute(
            self._state, self._target, self._t, self._success_integral
        )

        # Check termination
        terminated = self._check_terminated()
        truncated = self._step_count >= self._max_steps

        obs = self._build_observation()
        return obs, float(reward), terminated, truncated, {}

    def _check_terminated(self) -> bool:
        """Check termination conditions from Simulink system_226.xml."""
        h = self._state[6]
        alpha_deg = np.degrees(self._state[3])
        theta_deg = np.degrees(self._state[4])

        return (
            h <= -20.0
            or h >= 50.0
            or alpha_deg <= -50.0
            or alpha_deg >= 120.0
            or theta_deg <= -50.0
            or theta_deg >= 120.0
            or theta_deg >= 25.0
            or self._success_integral >= 1.0
        )
