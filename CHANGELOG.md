# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2026-05-24

### Added

- **Project Structure**: Complete Python project with `pyproject.toml`, YAML configs, and modular package layout
- **Data Conversion**: `scripts/convert_mat_to_npy.py` converts MATLAB `.mat`/`.xlsx` files to `.npy`/`.csv` format
- **Dynamics Model**: `src/tailsitter/dynamics.py` — 5-DOF longitudinal dynamics with RK4 integration, reverse-engineered from `mdl/Lon_mdl.slx`
- **Reward Function**: `src/tailsitter/reward.py` — 5-component reward (success, V tracking, θ tracking, α tracking, altitude rate), reverse-engineered from `RL/LonEnv.slx`
- **Gymnasium Environment**: `src/tailsitter/env.py` — 11D observation space, 2D action space, actuator dynamics, termination conditions
- **Normalization**: `src/tailsitter/normalization.py` — State/action normalization matching Simulink Gain blocks
- **Episode Reset**: `src/tailsitter/reset.py` — SAC (99/1 split) and PPO (50/50 split) reset modes
- **Training Scripts**: `scripts/train.py` and `scripts/train_continue.py` for SAC/PPO training with Stable-Baselines3
- **Evaluation**: `scripts/evaluate.py` and `src/tailsitter/plotting.py` for testing trained agents and visualization
- **Configuration**: YAML configs for physical parameters, normalization matrices, and hyperparameters
- **Tests**: 24 unit tests covering normalization, reward, and environment modules
