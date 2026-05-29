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

### Fixed

- **Trajectory Initial Guess**: Replaced linear interpolation with S-curve (cubic Hermite) for theta, parabolic pitch rate, and clamped throttle ramp — IPOPT now starts closer to feasibility
- **Trajectory us Bounds**: Relaxed `us` state bounds from `[0, 0]` to `[-0.01, 0.01]` to remove N+1 tight-bound equality constraints; `w_dus=10000` still drives us toward 0
- **Trajectory Elevator Dynamics**: Restored elevator sign inversion (`dcm(-de)`) and added CL/CD elevator increments (`dcL(-de)`, `dcD(-de)`) matching MATLAB `LonDyn.m`

### Changed

- **Trajectory IPOPT Tuning**: Added configurable `ipopt_mu_strategy`, `ipopt_hessian_approximation`, and `ipopt_warm_start` options to `TrajectoryOptConfig`
- **Trajectory Collocation**: Made collocation degree configurable (`collocation_degree=1,2,3`); default upgraded to d=2 (Radau) for better accuracy
- **Trajectory Two-Stage Solve**: Added `solve_from_guess()` and `resample_solution()` methods for warm-starting; CLI now supports `--two-stage` flag (coarse N=20 → fine N=num-nodes)
- **Trajectory CLI**: Added `--mu-strategy`, `--hessian`, `--collocation-degree`, and `--no-auto-trim` arguments
- **Trajectory Boundary Conditions**: Replaced hard equality constraints with penalty-based approach (`w_bc * sumsqr`) to avoid NLP infeasibility; continuation on w_bc (100→1e6) for progressive enforcement
- **Trajectory Objective Weights**: Normalized weights to O(1) for better NLP conditioning (was O(10⁴), now O(0.1))
- **Trajectory IPOPT Options**: Added `nlp_scaling_max_gradient=100`, MUMPS pivoting (`mumps_pivtol=1e-4`), `least_square_init_primal`, `acceptable_tol`, `bound_push/bound_frac`
- **Trajectory Trim Solver**: Added auto-trim mode (`auto_trim=True`) that computes boundary conditions from optimizer dynamics using `_find_trim()` with grid search over initial guesses
- **Trajectory Path Constraints**: Added alpha bounds (`-60°≤α≤120°`), theta upper bound (`θ≤120°`), tightened altitude bounds (`-20m≤h≤50m`)
- **Trajectory Initial Guess**: Improved S-curve interpolation for theta and pitch rate; added interior collocation state (Z) initial guess for d≥2
- **Trajectory Resample**: `resample_solution()` now supports `collocation_degree` parameter and generates Z state guesses for d≥2

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
