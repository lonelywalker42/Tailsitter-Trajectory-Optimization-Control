# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python project for training Reinforcement Learning agents (SAC/PPO) to control a tailsitter UAV's longitudinal dynamics during flight mode transitions (hover ↔ forward flight). Uses Stable-Baselines3 + Gymnasium. Refactored from a MATLAB/Simulink project.

## Prerequisites

- Python >= 3.10
- Install: `pip install -e .` (or `pip install -e ".[dev]"` for testing)

## Common Commands

```bash
# Convert MATLAB data files to .npy (one-time)
python scripts/convert_mat_to_npy.py

# Train SAC agent
python scripts/train.py --algo sac

# Train PPO agent
python scripts/train.py --algo ppo

# Continue training from checkpoint
python scripts/train_continue.py --checkpoint results/DP_sac_40000000/agent --algo sac --timesteps 20000000

# Evaluate trained agent
python scripts/evaluate.py --checkpoint results/DP_sac_40000000/agent --algo sac

# Trim analysis: single point
python scripts/trim.py point --V 15 --theta 10

# Trim analysis: level flight sweep
python scripts/trim.py level-flight --v-step 0.5

# Trim analysis: transition corridor sweep
python scripts/trim.py corridor --v-step 1 --theta-step 5

# Trim analysis: corridor with linearization
python scripts/trim.py corridor --v-step 2 --theta-step 10 --linearize

# Trajectory optimization
python scripts/trajectory_optimize.py --direction hover2forward

# Run tests
python -m pytest tests/ -v

# Run single test
python -m pytest tests/test_env.py::test_observation_space_shape -v
```

## Commit Conventions

Use semantic commit messages following the Conventional Commits specification:

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature or capability | `feat: add reward shaping for altitude tracking` |
| `fix` | Bug fix | `fix: correct elevator sign inversion in dynamics` |
| `refactor` | Code restructuring without behavior change | `refactor: extract aero lookup into separate method` |
| `docs` | Documentation only | `docs: update CLAUDE.md with commit conventions` |
| `perf` | Performance improvement | `perf: vectorize aerodynamic coefficient lookup` |
| `test` | Adding or updating tests | `test: add edge case for alpha=0 in dynamics` |
| `chore` | Build/CI/tooling changes | `chore: update pyproject.toml dependencies` |

## Development Workflow

### Feature Development

1. **Plan** — Create a plan file (use `/plan`) describing the feature, design decisions, and affected files
2. **Implement** — Write code following the architecture below
3. **Validate** — Run tests (`python -m pytest tests/ -v`), verify with `python -c "import tailsitter"`
4. **Commit** — Single focused commit with `feat:` prefix

### Refactoring

1. **Plan** — Describe what changes and why; identify affected files
2. **Implement** — Make structural changes without behavior changes
3. **Validate** — All existing tests must still pass
4. **Commit** — Single commit with `refactor:` prefix

### Bug Fixes

1. **Reproduce** — Write a failing test that demonstrates the bug
2. **Fix** — Minimal change to make the test pass
3. **Validate** — All tests pass
4. **Commit** — Single commit with `fix:` prefix

## Architecture

### Project Structure

```
TailsitterControl/
├── config/                    # YAML configuration files
│   ├── physical.yaml          # Physical parameters (mass, aero, prop)
│   ├── normalization.yaml     # State/action normalization matrices
│   ├── sac_hyperparams.yaml   # SAC training hyperparameters
│   └── ppo_hyperparams.yaml   # PPO training hyperparameters
├── data/
│   ├── raw/                   # Original .mat/.xlsx files from MATLAB
│   └── processed/             # Converted .npy files
├── docs/                      # Supplementary documentation
│   ├── theory.md              # Theoretical foundations (dynamics, RL, reward)
│   ├── architecture.md        # Design decisions and technical details
│   ├── user_guide.md          # Usage instructions for end users
│   └── development.md         # Development workflow and milestones
├── matlab/                    # Original MATLAB/Simulink code (reference)
│   ├── RL/                    # RL training scripts and Simulink environment
│   │   ├── train_script.m     # SAC training (with Ctrl+C save)
│   │   ├── train_continue.m   # Resume training from checkpoint
│   │   ├── train_callback.m   # Custom terminal progress display
│   │   ├── trian_PPO_script.m # PPO training script
│   │   ├── initialize.m       # Parameter and data initialization
│   │   ├── LonEnv.slx         # Simulink RL environment model
│   │   ├── agent_test.slx     # Simulink agent test model
│   │   └── result/            # Training result processing scripts
│   └── mdl/                   # Plant model and aerodynamic data
│       ├── Lon_mdl.slx        # Simulink longitudinal dynamics model
│       ├── aero_cfg*.mat      # Aerodynamic coefficient tables
│       ├── aerosurface.mat    # Elevator increment data
│       └── prop.mat           # Propulsion thrust lookup
├── scripts/                   # Training and evaluation entry points
│   ├── convert_mat_to_npy.py  # One-time data conversion
│   ├── train.py               # Main training script
│   ├── train_continue.py      # Continue from checkpoint
│   ├── evaluate.py            # Test trained agent
│   ├── trajectory_optimize.py # CasADi trajectory optimization
│   └── trim.py                # Trim analysis and transition corridor
├── src/tailsitter/            # Core Python package
│   ├── config.py              # YAML config loaders, dataclasses
│   ├── dynamics.py            # 5-DOF longitudinal dynamics
│   ├── env.py                 # Gymnasium environment
│   ├── linearization.py       # Linearization and eigenvalue analysis
│   ├── normalization.py       # State/action normalization
│   ├── plotting.py            # Matplotlib visualization
│   ├── reset.py               # Episode reset logic
│   ├── reward.py              # 5-component reward function
│   ├── trajectory_optimization.py # CasADi trajectory optimization
│   ├── trajectory_plotting.py # Trajectory result visualization
│   ├── trim.py                # Trim solvers and corridor sweep
│   └── trim_plotting.py       # Trim corridor visualization
└── tests/                     # Unit tests
    ├── test_env.py
    ├── test_normalization.py
    ├── test_reward.py
    └── test_trim.py
```

### Core Modules (`src/tailsitter/`)

- `config.py` — YAML config loaders, dataclasses for all parameters
- `dynamics.py` — 5-DOF longitudinal dynamics with RK4 integration (reverse-engineered from `mdl/Lon_mdl.slx`)
- `reward.py` — 5-component reward function (reverse-engineered from `RL/LonEnv.slx`)
- `env.py` — Gymnasium environment wrapping dynamics + reward + normalization
- `normalization.py` — State/action normalization matching Simulink Gain blocks
- `reset.py` — Episode reset logic (SAC: 99/1 split, PPO: 50/50 split)
- `plotting.py` — Matplotlib visualization (reward curves, simulation response)
- `trim.py` — Trim solvers and transition corridor sweep (refactored from `matlab/trim/`)
- `linearization.py` — Linearization and eigenvalue/controllability analysis (refactored from `matlab/trim/linearization/`)
- `trim_plotting.py` — Corridor and eigenvalue heatmap visualization
- `trajectory_optimization.py` — CasADi + IPOPT trajectory optimization
- `trajectory_plotting.py` — Trajectory result visualization

### State Space

- **Plant state (7D)**: `[u, w, V, alpha, theta, q, h]`
- **Observation (11D)**: `[V, alpha, theta, q, h, dt_last, de_last, dV, dalpha, dtheta, cmd]`
- Observation is normalized by `NormMatrixState` before being sent to the agent

### Action Space (2D)

`[dt, de]` in [-1, 1], denormalized to physical units via `inv(NormMatrixAct) @ action + BiasAct`

### Dynamics Model

The ODE in `dynamics.py` was extracted from Stateflow scripts inside `mdl/Lon_mdl.slx`:

- Force equation: aero (L,D) → body frame + propulsion (2×Fprop) + gravity
- State derivatives: 5-DOF with moving mass coupling (simplified: state_s=[0,0,0])
- Aero coefficients: lookup tables from `aero_cfg2.mat` + elevator increments from `aerosurface.mat`
- Propulsion: throttle → thrust lookup from `prop.mat`

### Reward Function (from `RL/LonEnv.slx` XML)

```
f(x) = 1/sqrt(0.1*x² + 1)
reward = -0.05*exp(0.1*t) + 1000*success + f(5*|ΔV|) + f(|Δθ°|) + f(|Δα°|) + 0.05*(dh/dt)²
```

### Termination Conditions

h ≤ -20, h ≥ 50, |α| ≥ 120°, |θ| ≥ 120°, θ ≥ 25°, success_integral ≥ 1, t ≥ 20s

## Documentation Sync Rules

When making changes, update documentation as follows:

| Change Type | Update Required |
|-------------|-----------------|
| `feat` | CHANGELOG.md (add entry), docs/architecture.md (if design changes), docs/user_guide.md (if user-facing) |
| `fix` | CHANGELOG.md (add entry) |
| `refactor` | docs/architecture.md (if structure changes) |
| `docs` | Update the specific doc file only |
| `perf` | CHANGELOG.md (add entry) |

## Notes

- Original MATLAB code is preserved in `matlab/` (subdirs: `RL/` for training scripts, `mdl/` for plant model and aero data)
- MATLAB training scripts support Ctrl+C interruption with automatic checkpoint saving (`checkpoint_interrupt_*.mat`)
- Resume interrupted training with `train_continue.m` — it auto-loads the latest checkpoint
- Custom terminal progress display via `train_callback.m` (shows episode reward, avg reward, progress bar, ETA)
- When providing new MATLAB code for Python refactoring, place it under `matlab/` following the existing structure
- Dynamics model uses Numba JIT (`@numba.njit`) for the RK4 integration kernel — first call compiles (~2-5s), cached thereafter
- Lookup tables use `np.interp` inside Numba (replaces `scipy.interp1d`)
- Multi-environment parallelism: set `n_envs` in YAML config or `--n-envs` CLI arg (default: 1)
- Actuator model: first-order lag (τ=0.001s) with elevator sign inversion
- RNG seed: 77777 (configurable in `config/physical.yaml`)
