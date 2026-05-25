"""Main training script for SAC/PPO agents.

Replaces RL/train_script.m and RL/trian_PPO_script.m.

Usage:
    python scripts/train.py --algo sac
    python scripts/train.py --algo ppo
    python scripts/train.py --algo sac --n-envs 4
    python scripts/train.py --algo sac --config-dir config --results-dir results
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tailsitter.config import (
    load_env_config,
    load_normalization_config,
    load_physical_config,
    load_training_config,
    ResetConfig,
)
from tailsitter.callbacks import ProgressCallback
from tailsitter.dynamics import LongitudinalDynamics
from tailsitter.env import TailsitterLongitudinalEnv
from tailsitter.normalization import Normalizer
from tailsitter.reset import EpisodeResetter
from tailsitter.reward import RewardCalculator


def load_aero_data(data_dir: Path) -> dict:
    """Load aerodynamic data from processed .npy files."""
    aero = {}
    # Aero coefficients (cfg2 = baseline)
    for key in ["alpha", "cL", "cD", "cm"]:
        fname = f"aero_cfg2_{key}.npy"
        path = data_dir / fname
        if path.exists():
            aero[key] = np.load(str(path))

    # Surface increments
    for key in ["ele", "dcL", "dcD", "dcm"]:
        fname = f"aerosurface_{key}.npy"
        path = data_dir / fname
        if path.exists():
            aero[key] = np.load(str(path))

    # Propulsion
    for key in ["throttle", "thrust"]:
        fname = f"prop_{key}.npy"
        path = data_dir / fname
        if path.exists():
            aero[key] = np.load(str(path))

    return aero


def _make_single_env(phys, env_cfg, norm_cfg, aero_data, reset_mode):
    """Create a single environment instance (used by make_vec_env)."""
    dynamics = LongitudinalDynamics(phys, aero_data)
    reward_calc = RewardCalculator(env_cfg)
    normalizer = Normalizer(norm_cfg)
    resetter = EpisodeResetter(ResetConfig())

    return TailsitterLongitudinalEnv(
        dynamics=dynamics,
        reward_calc=reward_calc,
        normalizer=normalizer,
        resetter=resetter,
        env_config=env_cfg,
        phys_config=phys,
        reset_mode=reset_mode,
    )


def create_env(config_dir: Path, data_dir: Path, reset_mode: str,
               n_envs: int = 1):
    """Create the Gymnasium environment (single or vectorized).

    Args:
        config_dir: Path to config directory
        data_dir: Path to processed data directory
        reset_mode: "sac" or "ppo"
        n_envs: Number of parallel environments (1 = single env)

    Returns:
        gym.Env or VecEnv
    """
    phys = load_physical_config(config_dir)
    env_cfg = load_env_config(config_dir)
    norm_cfg = load_normalization_config(config_dir)
    aero_data = load_aero_data(data_dir)

    if n_envs == 1:
        return _make_single_env(phys, env_cfg, norm_cfg, aero_data, reset_mode)

    from stable_baselines3.common.env_util import make_vec_env

    return make_vec_env(
        lambda: _make_single_env(phys, env_cfg, norm_cfg, aero_data, reset_mode),
        n_envs=n_envs,
    )


def main():
    parser = argparse.ArgumentParser(description="Train SAC/PPO agent for tailsitter control")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"],
                        help="Algorithm to use")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Path to config directory")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to results directory")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override number of parallel environments")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    results_dir = Path(args.results_dir)
    data_dir = config_dir.parent / "data" / "processed"

    # Load training config
    train_cfg = load_training_config(config_dir, args.algo)
    total_timesteps = args.timesteps or train_cfg.total_timesteps
    n_envs = args.n_envs if args.n_envs is not None else train_cfg.n_envs

    # Create environment
    env = create_env(config_dir, data_dir, reset_mode=args.algo, n_envs=n_envs)

    # Create SB3 agent
    from stable_baselines3 import SAC, PPO
    from torch import nn

    activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[train_cfg.activation_fn]

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.learning_rate,
            buffer_size=train_cfg.buffer_size,
            batch_size=train_cfg.batch_size,
            tau=train_cfg.tau,
            gamma=train_cfg.gamma,
            train_freq=train_cfg.train_freq,
            gradient_steps=train_cfg.gradient_steps,
            learning_starts=train_cfg.learning_starts,
            policy_kwargs=dict(
                net_arch=train_cfg.net_arch,
                activation_fn=activation_fn,
            ),
            verbose=0,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            ent_coef=train_cfg.ent_coef,
            max_grad_norm=train_cfg.max_grad_norm,
            vf_coef=train_cfg.vf_coef,
            policy_kwargs=dict(
                net_arch=train_cfg.net_arch,
                activation_fn=activation_fn,
            ),
            verbose=0,
        )

    # Create results directory
    run_name = f"DP_{args.algo}_{total_timesteps}"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {args.algo.upper()} for {total_timesteps} timesteps")
    print(f"Parallel environments: {n_envs}")
    print(f"Results will be saved to: {run_dir}\n")

    # Train with Ctrl+C checkpoint saving
    callback = ProgressCallback(total_timesteps=total_timesteps)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        pass

    # Save (always — normal completion or Ctrl+C interruption)
    model.save(str(run_dir / "agent"))
    print(f"\nAgent saved to {run_dir / 'agent.zip'}")


if __name__ == "__main__":
    main()
