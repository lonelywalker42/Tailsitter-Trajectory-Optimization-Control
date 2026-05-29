# User Guide

This guide explains how to use the TailsitterControl project to train and evaluate RL agents for tailsitter UAV control.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TailsitterControl

# Install in development mode
pip install -e ".[dev]"

# Convert MATLAB data files (one-time)
python scripts/convert_mat_to_npy.py
```

## Training

### Train a SAC Agent

```bash
python scripts/train.py --algo sac
```

This trains a SAC agent for 40 million timesteps (default) and saves checkpoints to `results/`.

### Train a PPO Agent

```bash
python scripts/train.py --algo ppo
```

### Custom Training Parameters

```bash
# Train for 20 million timesteps
python scripts/train.py --algo sac --timesteps 20000000

# Use custom config directory
python scripts/train.py --algo sac --config-dir my_config/

# Save to custom directory
python scripts/train.py --algo sac --results-dir my_results/
```

### Continue Training from Checkpoint

```bash
python scripts/train_continue.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac \
    --timesteps 20000000
```

## Evaluation

### Evaluate a Trained Agent

```bash
python scripts/evaluate.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac
```

This runs two test scenarios:
1. **Forward transition**: Hover → Forward flight
2. **Back transition**: Forward flight → Hover

Results are saved to `results/evaluation/` as PNG plots.

### Understanding the Plots

The evaluation generates a 4×3 subplot grid for each transition:

| Plot | Description |
|------|-------------|
| V | Airspeed over time |
| α | Angle of attack over time |
| θ | Pitch angle over time |
| q | Pitch rate over time |
| h | Altitude over time |
| dt | Throttle command over time |
| de | Elevator command over time |
| dV | Airspeed error over time |
| dα | Alpha error over time |
| dθ | Theta error over time |
| V-θ | Phase portrait (V vs θ) |

## Configuration

### Physical Parameters (`config/physical.yaml`)

Contains all physical constants:
- Mass properties (mb, ms, m, Jy)
- Aerodynamic reference areas (S, c)
- Propulsion parameters (R, rho)
- Actuator dynamics (tau)
- Environment settings (Ts, Tf)

### Normalization (`config/normalization.yaml`)

Defines the normalization matrices:
- `state_norm_diag`: Diagonal of the state normalization matrix
- `action_norm_diag`: Diagonal of the action normalization matrix
- `action_bias`: Bias vector for action denormalization

### Hyperparameters

**SAC** (`config/sac_hyperparams.yaml`):
- `gamma`: Discount factor (0.99)
- `tau`: Soft update coefficient (1e-3)
- `lr`: Learning rate (1e-3)
- `buffer_size`: Replay buffer size (100000)
- `batch_size`: Training batch size (512)
- `learning_starts`: Steps before training (256)

**PPO** (`config/ppo_hyperparams.yaml`):
- `gamma`: Discount factor (0.99)
- `gae_lambda`: GAE lambda (0.95)
- `clip_range`: PPO clip range (0.2)
- `ent_coef`: Entropy coefficient (0.01)
- `n_steps`: Steps per update (1024)
- `batch_size`: Mini-batch size (512)
- `n_epochs`: Epochs per update (3)

## Flight Modes

The system supports two flight modes controlled by the `cmd` variable:

| cmd | Mode | Target State |
|-----|------|--------------|
| -1 | Forward flight | V=20 m/s, α≈5°, θ≈5° |
| 1 | Hover | V≈0 m/s, α≈90°, θ≈90° |

## Trim Analysis

The trim module finds equilibrium points (steady-state control inputs) across the flight envelope and computes the transition corridor.

### Single Point Trim

Find trim at a specific operating point:

```bash
python scripts/trim.py point --V 15 --theta 10
```

Output includes throttle, elevator deflection, angle of attack, flight path angle, and residual.

### Level Flight Trim

Sweep level flight trim (θ=α, γ=0) over velocity range:

```bash
python scripts/trim.py level-flight --v-step 0.5
```

Generates plots of θ, δe, and δt vs airspeed in `results/trim/`.

### Transition Corridor

Compute the full transition corridor over a (V, θ) grid:

```bash
python scripts/trim.py corridor --v-step 1 --theta-step 5
```

Generates 7 plots:
1. Corridor boundary (trimmable region)
2. Trim angle of attack heatmap
3. Flight path angle heatmap
4. Elevator deflection heatmap
5. Throttle heatmap
6. Forward transition residual moment
7. Backward transition residual moment

### Corridor with Linearization

Add eigenvalue and controllability analysis:

```bash
python scripts/trim.py corridor --v-step 2 --theta-step 10 --linearize
```

Generates additional plots for eigenvalue real parts, real eigenvalue count, and controllability Gramian.

### Custom Configuration

```bash
# Use aero config 1 with moving mass at -0.2m
python scripts/trim.py corridor --config aero_cfg1 --xs -0.2

# Fine grid with parallel processing
python scripts/trim.py corridor --v-step 0.5 --theta-step 2 --n-jobs 4

# Save to custom directory
python scripts/trim.py corridor --output-dir results/my_trim
```

### Understanding Trim Results

| Output | Description |
|--------|-------------|
| `trim_flag` | 1 = trimmable, 0 = not trimmable |
| `throttle` | Trim throttle command [0, 1] |
| `elevator` | Trim elevator deflection [deg] |
| `alpha` | Trim angle of attack [deg] |
| `gamma` | Flight path angle θ-α [deg] |
| `max_moment_ft` | Forward transition residual moment |
| `max_moment_bt` | Backward transition residual moment |

## Trajectory Optimization

Find optimal transition trajectories using CasADi + IPOPT:

```bash
# Forward flight → hover (default: auto-trim, d=2 collocation)
python scripts/trajectory_optimize.py --direction forward2hover

# Hover → forward flight
python scripts/trajectory_optimize.py --direction hover2forward

# Custom settings
python scripts/trajectory_optimize.py --num-nodes 200 --tf-guess 15

# Two-stage solve: coarse (N=20) then fine (warm-started)
python scripts/trajectory_optimize.py --direction forward2hover --two-stage

# L-BFGS Hessian with monotone barrier strategy
python scripts/trajectory_optimize.py --direction forward2hover --hessian limited-memory --mu-strategy monotone

# Use hardcoded boundary conditions instead of auto-trim
python scripts/trajectory_optimize.py --direction forward2hover --no-auto-trim

# Backward Euler collocation (faster, less accurate)
python scripts/trajectory_optimize.py --direction forward2hover --collocation-degree 1
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--direction` | `forward2hover` | Transition direction |
| `--num-nodes` | `100` | Number of mesh nodes |
| `--tf-guess` | `10.0` | Initial guess for transition time (seconds) |
| `--ipopt-max-iter` | `3000` | IPOPT maximum iterations |
| `--two-stage` | off | Two-stage solve: coarse → fine (warm-started) |
| `--mu-strategy` | `adaptive` | IPOPT barrier update strategy (`adaptive` or `monotone`) |
| `--hessian` | `exact` | Hessian approximation (`exact` or `limited-memory`) |
| `--collocation-degree` | `2` | Radau collocation degree (1, 2, or 3) |
| `--no-auto-trim` | off | Use hardcoded boundary conditions instead of auto-trim |

### Auto-Trim Mode

By default, the optimizer computes boundary conditions from its own dynamics model using `_find_trim()`. This avoids model mismatch between the trim solver and the optimizer dynamics (which include induced velocity effects). The auto-trim solver:

1. Finds forward flight trim at V=20 m/s (typically θ≈58°, γ≈60° — climbing flight)
2. Finds hover trim at V=0.01 m/s (θ=90°)
3. Uses grid search over initial guesses to find the best equilibrium

Use `--no-auto-trim` to use the original hardcoded boundary conditions from `TrajectoryOptConfig`.

### Output

Results are saved to `results/trajopt/` as:
- `.npz` file with time, state, control, cost, and solve time
- State trajectory plot (V-θ phase portrait + time histories)
- Control input plot (throttle, elevator rates)
- Angle of attack plot

### Known Issues

The trajectory optimizer with induced velocity dynamics may not fully satisfy boundary conditions. The optimizer converges (IPOPT reports "Optimal Solution Found") but terminal states may deviate from targets. This is due to the stiff, nonlinear nature of the induced velocity dynamics model, not a solver configuration issue. See `docs/architecture.md` for detailed analysis.

## DF-Based Trajectory Optimization

An alternative trajectory optimization approach using differential flatness. Instead of optimizing in the full 11D state space, this operates in the 2D flat output space (V, γ), which is more robust and faster.

### DF Corridor Sweep

Compute the DF-based transition corridor:

```bash
python scripts/df_trajectory.py df-trim

# Custom grid
python scripts/df_trajectory.py df-trim --v-min 0 --v-max 20 --dv 0.5 --dgamma 1
```

Generates 6 heatmap plots: α, θ, q, thrust, elevator, and q_max over the (V, γ) corridor.

### DF Trajectory Optimization

Find optimal transition trajectories using the DF approach:

```bash
# Hover → forward flight
python scripts/df_trajectory.py df-optimize --direction hover2forward

# Forward flight → hover
python scripts/df_trajectory.py df-optimize --direction forward2hover

# Custom weights
python scripts/df_trajectory.py df-optimize --w-time 0.2 --w-safety 0.3

# Custom velocity grid resolution
python scripts/df_trajectory.py df-optimize --dv 0.5
```

Output includes:
- 3×3 subplot: trajectory, V, γ, α/θ, V̇, γ̇, q, throttle, elevator
- `.npz` file with all trajectory data

### Baseline Corridor Path

Compute the baseline geometric path within the feasibility corridor:

```bash
python scripts/df_trajectory.py df-baseline --direction hover2forward

# Custom control points
python scripts/df_trajectory.py df-baseline --n-points 20
```

### Trajectory Comparison

Run both proposed (DF optimization) and baseline (corridor path) and generate comparison:

```bash
python scripts/df_trajectory.py df-compare --direction hover2forward
```

Generates:
- DF trajectory plot (3×3 subplot)
- Comparison plot: both trajectories on the corridor feasibility map with pitch rate margin background

### DF Trajectory Simulation

Simulate trajectory tracking and analyze errors (replaces Simulink-based simulation):

```bash
# Simulate hover → forward flight transition
python scripts/df_trajectory.py df-simulate --direction hover2forward

# Simulate forward flight → hover transition
python scripts/df_trajectory.py df-simulate --direction forward2hover

# Custom timestep and initial altitude
python scripts/df_trajectory.py df-simulate --direction hover2forward --dt 0.01 --h0 10.0
```

Output includes:
- Error analysis tables (RMSE, max deviation, end-point error) for both proposed and baseline trajectories
- Performance grading (excellent/good/fair/poor) matching MATLAB thresholds
- 3×2 error analysis plots showing reference vs actual tracking for V, γ, θ, q, h, α
- `.npz` file with all simulation data

### DF CLI Arguments

**Global arguments** (apply to all subcommands):

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `aero_cfg2` | Aero configuration name |
| `--output-dir` | `results/df_trajectory` | Output directory |

**df-optimize / df-compare arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dv` | `1.0` | Velocity grid step [m/s] |
| `--dgamma` | `1.0` | FPA grid step [deg] |
| `--w-time` | `0.1` | Time cost weight |
| `--w-energy` | `0.1` | Control energy weight |
| `--w-safety` | `0.1` | Pitch rate safety margin weight |
| `--margin-k` | `2.0` | Pitch rate margin proportionality coefficient |

**df-simulate arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--direction` | `hover2forward` | Transition direction |
| `--dt` | `0.02` | Simulation timestep [s] |
| `--h0` | `0.0` | Initial altitude [m] |
| `--acc` | `5.0` | Baseline assumed acceleration [m/s²] |
| `--n-points` | `16` | Number of baseline control points |

### DF vs CasADi Comparison

| Aspect | DF Approach | CasADi Approach |
|--------|-------------|-----------------|
| State space | 2D (V, γ) | 11D (full dynamics) |
| Solver | scipy trust-constr | IPOPT |
| Speed | Fast (~seconds) | Slow (~minutes) |
| Robustness | More robust | Sensitive to initial guess |
| Dynamics fidelity | Flatness transfer | Full 5-DOF with induced velocity |
| Best for | Quick exploration, corridor analysis | High-fidelity trajectory design |

## Troubleshooting

### ModuleNotFoundError: No module named 'tailsitter'

Install the package in development mode:
```bash
pip install -e .
```

### Missing data files

Run the data conversion script:
```bash
python scripts/convert_mat_to_npy.py
```

### Training not converging

1. Check that data files exist in `data/processed/`
2. Verify hyperparameters in config files
3. Monitor reward curves in TensorBoard:
   ```bash
   tensorboard --logdir results/
   ```
