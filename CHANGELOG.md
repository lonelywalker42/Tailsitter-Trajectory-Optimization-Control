# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Trim Analysis**: `src/tailsitter/trim.py` — trim solvers and transition corridor sweep, refactored from `matlab/trim/`. Includes level flight trim, general trim with random restarts (SLSQP), and max-residual-moment optimization with force constraints
- **Linearization**: `src/tailsitter/linearization.py` — numerical Jacobian computation, eigenvalue/controllability/stability/modal analysis at trim points, refactored from `matlab/trim/linearization/`
- **Trim Plotting**: `src/tailsitter/trim_plotting.py` — SCI-style corridor heatmaps (alpha, gamma, elevator, throttle, residual moment) and eigenvalue corridor plots
- **Trim CLI**: `scripts/trim.py` — command-line entry point with `corridor`, `level-flight`, and `point` subcommands
- **Trim Tests**: `tests/test_trim.py` — 18 unit tests covering aero computation, trim solvers, corridor sweep, and linearization
- **Trajectory Optimization**: `src/tailsitter/trajectory_optimization.py` — CasADi direct collocation optimizer with IPOPT backend for finding optimal transition trajectories (forward flight ↔ hover)
- **Trajectory Plotting**: `src/tailsitter/trajectory_plotting.py` — visualization module for trajectory optimization results
- **Trajectory CLI**: `scripts/trajectory_optimize.py` — command-line entry point for trajectory optimization
- **Trajectory Tests**: `tests/test_trajectory_optimization.py` — unit tests for config, interpolants, collocation coefficients, dynamics, and NLP construction
- **CasADi Dependency**: Added `casadi>=3.6` to `pyproject.toml` and `requirements.txt`

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
