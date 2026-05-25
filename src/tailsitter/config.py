"""Configuration management for the tailsitter control project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class PhysicalConfig:
    """Physical parameters of the tailsitter UAV."""
    mb: float = 6.3
    ms: float = 0.2
    m: float = 6.5
    mu: float = 0.0308  # ms/m
    Jy: float = 0.183
    g: float = 9.81
    S: float = 0.62
    c: float = 0.31
    prop_R: float = 0.2032
    rho: float = 1.225
    actuator_tau: float = 0.001
    elevator_sign: int = -1


@dataclass
class EnvConfig:
    """Environment configuration."""
    Ts: float = 0.02
    Tf: float = 20.0
    obs_dim: int = 11
    act_dim: int = 2
    seed: int = 77777


@dataclass
class NormalizationConfig:
    """State/action normalization parameters."""
    state_norm_diag: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.1, 0.05, 0.05, 0.1, 0.5, 1.0, 0.05, 0.2, 0.1, 0.1, 1.0]
        )
    )
    action_norm_diag: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 0.04])
    )
    action_bias: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, 5.0])
    )

    @property
    def state_norm_matrix(self) -> np.ndarray:
        return np.diag(self.state_norm_diag)

    @property
    def action_norm_matrix(self) -> np.ndarray:
        return np.diag(self.action_norm_diag)

    @property
    def action_inv_norm_matrix(self) -> np.ndarray:
        return np.diag(1.0 / self.action_norm_diag)


@dataclass
class ResetConfig:
    """Episode reset configuration."""
    sac_forward_prob: float = 0.01
    sac_randomize_initial: bool = True
    ppo_forward_prob: float = 0.5
    ppo_randomize_initial: bool = False

    hover_state: tuple = (0.01, np.pi / 2, np.pi / 2)
    hover_initial: tuple = (0.0, 0.01, np.pi / 2, 0.0, 0.0)
    forward_state: tuple = (20.0, 4.9184 * np.pi / 180, 4.9184 * np.pi / 180)
    forward_initial: tuple = (19.926, 1.715, 4.9184 * np.pi / 180, 0.0, 0.0)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    algorithm: str = "SAC"
    gamma: float = 0.99
    tau: float = 0.001
    learning_rate: float = 0.001
    buffer_size: int = 100000
    batch_size: int = 512
    warmup_steps: int = 256
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 256
    total_timesteps: int = 40000000
    save_reward_threshold: float = 8000.0
    score_window: int = 20

    # PPO-specific
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_epochs: int = 3
    n_steps: int = 1024
    max_grad_norm: float = 1.0
    vf_coef: float = 0.5

    # Network
    net_arch: list = field(default_factory=lambda: [256, 256])
    activation_fn: str = "relu"


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_physical_config(config_dir: str | Path) -> PhysicalConfig:
    """Load physical parameters from YAML."""
    data = load_yaml(Path(config_dir) / "physical.yaml")
    return PhysicalConfig(
        mb=data["mass"]["body"],
        ms=data["mass"]["moving"],
        m=data["mass"]["total"],
        mu=data["mass"]["moving"] / data["mass"]["total"],
        Jy=data["inertia"]["Jy"],
        g=data["gravity"],
        S=data["reference"]["area"],
        c=data["reference"]["chord"],
        prop_R=data["propeller"]["radius"],
        rho=data["air_density"],
        actuator_tau=data["actuator"]["tau"],
        elevator_sign=data["actuator"]["elevator_sign"],
    )


def load_env_config(config_dir: str | Path) -> EnvConfig:
    """Load environment config from physical.yaml."""
    data = load_yaml(Path(config_dir) / "physical.yaml")
    return EnvConfig(
        Ts=data["environment"]["Ts"],
        Tf=data["environment"]["Tf"],
    )


def load_normalization_config(config_dir: str | Path) -> NormalizationConfig:
    """Load normalization parameters from YAML."""
    data = load_yaml(Path(config_dir) / "normalization.yaml")
    return NormalizationConfig(
        state_norm_diag=np.array(data["state_norm_diag"]),
        action_norm_diag=np.array(data["action_norm_diag"]),
        action_bias=np.array(data["action_bias"]),
    )


def load_training_config(config_dir: str | Path, algorithm: str) -> TrainingConfig:
    """Load training hyperparameters from YAML."""
    algo = algorithm.lower()
    path = Path(config_dir) / f"{algo}_hyperparams.yaml"
    data = load_yaml(path)
    return TrainingConfig(**{k: v for k, v in data.items() if k in TrainingConfig.__dataclass_fields__})
