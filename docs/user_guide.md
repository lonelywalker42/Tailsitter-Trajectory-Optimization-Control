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
# Forward flight → hover
python scripts/trajectory_optimize.py --direction forward2hover

# Hover → forward flight
python scripts/trajectory_optimize.py --direction hover2forward

# Custom settings
python scripts/trajectory_optimize.py --num-nodes 200 --tf-guess 15
```

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
