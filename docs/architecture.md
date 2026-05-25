# Architecture

This document describes the technical architecture and design decisions of the TailsitterControl project.

## Overview

TailsitterControl is a reinforcement learning system for controlling a tailsitter UAV's longitudinal dynamics. The system trains SAC/PPO agents to perform flight mode transitions between hover and forward flight.

## Design Decisions

### 1. Gymnasium + Stable-Baselines3

**Decision**: Use Gymnasium for the environment interface and Stable-Baselines3 for RL algorithms.

**Rationale**:
- Gymnasium is the standard RL environment interface, compatible with most RL libraries
- Stable-Baselines3 provides well-tested SAC/PPO implementations with good documentation
- Both are actively maintained and widely used in the RL community

### 2. YAML Configuration

**Decision**: Use YAML files for all configurable parameters instead of Python constants.

**Rationale**:
- Separates configuration from code
- Easy to modify hyperparameters without code changes
- Supports multiple experiment configurations by swapping config files
- Matches the original MATLAB workspace variable approach

### 3. Modular Package Structure

**Decision**: Separate dynamics, reward, normalization, and reset into distinct modules.

**Rationale**:
- Each module has a single responsibility
- Modules can be tested independently
- Easy to swap components (e.g., different reward functions)
- Mirrors the original Simulink block diagram structure

### 4. RK4 Integration

**Decision**: Use 4th-order Runge-Kutta for dynamics integration.

**Rationale**:
- Matches the original Simulink solver behavior
- Provides sufficient accuracy for the 0.02s timestep
- More stable than Euler integration for stiff dynamics

### 5. Lookup Table Interpolation

**Decision**: Use `scipy.interpolate.interp1d` for aerodynamic coefficient lookup.

**Rationale**:
- Linear interpolation matches Simulink lookup table behavior
- Handles the discrete aerodynamic data from `.mat` files
- Efficient for repeated evaluations during RL training

## Data Flow

```
┌─────────────┐     action      ┌─────────────┐
│  RL Agent   │────────────────►│  env.py     │
│  (SB3)      │                 │             │
└─────────────┘                 │  ┌─────────┐│
       ▲                        │  │normalize││
       │  obs                   │  │  .py    ││
       │                        │  └────┬────┘│
       │                        │       │     │
       │                        │  ┌────▼────┐│
       │                        │  │dynamics ││
       │                        │  │  .py    ││
       │                        │  └────┬────┘│
       │                        │       │     │
       │                        │  ┌────▼────┐│
       └────────────────────────│  │ reward  ││
                                │  │  .py    ││
                                │  └─────────┘│
                                └─────────────┘
```

### Step Execution

1. Agent selects action `a ∈ [-1, 1]²`
2. `normalization.py` denormalizes action to physical units
3. `dynamics.py` integrates ODE with RK4, returns new state
4. `reward.py` computes reward from state and target
5. `env.py` builds 11D observation, checks termination
6. Observation returned to agent

## State Representation

### Plant State (7D)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | u | Body x-axis velocity | m/s |
| 1 | w | Body z-axis velocity | m/s |
| 2 | V | Airspeed | m/s |
| 3 | α | Angle of attack | rad |
| 4 | θ | Pitch angle | rad |
| 5 | q | Pitch rate | rad/s |
| 6 | h | Altitude | m |

### Observation (11D)

| Index | Symbol | Description |
|-------|--------|-------------|
| 0 | V | Airspeed |
| 1 | α | Angle of attack |
| 2 | θ | Pitch angle |
| 3 | q | Pitch rate |
| 4 | h | Altitude |
| 5 | dt_last | Last throttle command |
| 6 | de_last | Last elevator command |
| 7 | dV | Airspeed error |
| 8 | dα | Alpha error |
| 9 | dθ | Theta error |
| 10 | cmd | Flight mode command |

## Dynamics Model

The dynamics model was reverse-engineered from `mdl/Lon_mdl.slx` by extracting Stateflow scripts from the `.slx` ZIP archive.

### Force Computation

```
Fx = sin(α)·L - cos(α)·D + 2·Fprop - m·g·sin(θ)
Fz = -cos(α)·L - sin(α)·D + m·g·cos(θ)
```

### State Derivatives

```
q̇  = (my - ms·xs·g·cos(θ) - 2·ms·xs·vs·q) / (Jy + ms·xs²)
u̇  = Fx/m - μ·vsdot
ẇ  = Fz/m - μ·(-q̇·xs - q·vs)
θ̇  = q
ḣ  = w·cos(θ) - u·sin(θ)
```

### Aerodynamic Model

1. Dynamic pressure: `Q = 0.5·ρ·S·V²`
2. Base coefficients: `cL(α)`, `cD(α)`, `cm(α)` from lookup tables
3. Elevator increments: `dcL(δe)`, `dcD(δe)`, `dcm(δe)` from lookup tables
4. Forces: `L = Q·CL`, `D = Q·CD`, `my = Q·c·Cm`

## Reward Function

The reward function was reverse-engineered from `RL/LonEnv.slx` XML.

```
f(x) = 1/√(0.1·x² + 1)

r_success = 1000 · success_integral
r_V = f(5·|V - V_target|)
r_θ = f(|θ_deg - θ_target_deg|)
r_α = f(|α_deg - α_target_deg|)
r_dh = 0.05·(dh/dt)²
time_coef = -0.05·exp(0.1·t)

reward = time_coef + r_success + r_V + r_θ + r_α + r_dh
```

### Success Integral

Accumulates when the normalized tracking error is below threshold:

```
error = ‖[0.5·ΔV, 0.5·57.3·Δα, 0.5·57.3·Δθ]‖₂
if error < 2.0:
    success_integral += Ts / 20.0
```

Clamped to [0, 1]. When `success_integral ≥ 1`, episode terminates successfully.

## Termination Conditions

| Condition | Threshold | Source |
|-----------|-----------|--------|
| Ground crash | h ≤ -20 m | Simulink |
| Too high | h ≥ 50 m | Simulink |
| Extreme alpha | \|α\| ≥ 120° | Simulink |
| Extreme theta | \|θ\| ≥ 120° | Simulink |
| Excessive pitch | θ ≥ 25° | Simulink |
| Success | success_integral ≥ 1 | Simulink |
| Timeout | t ≥ 20 s | Config |

## Trajectory Optimization

The trajectory optimization module (`trajectory_optimization.py`) finds optimal transition trajectories using CasADi + IPOPT, independent of the RL pipeline.

### Design

- **Solver**: CasADi `nlpsol` with IPOPT backend
- **Collocation**: Configurable Radau degree (d=1 backward Euler, d=2, d=3); default d=1 for reliable convergence
- **Dynamics**: 11D state (8 plant + 3 augmented controls), 3D control (rates), matching MATLAB `LonDyn.m` cfg1 tables
- **Aero model**: CL/CD with elevator increments (`dcL(-de)`, `dcD(-de)`), elevator sign inversion matching MATLAB
- **Objective**: `min(tf + ∫ L(x,u) dt)` with weights matching MATLAB `UAVContinuous.m`
- **Two-stage solve**: Coarse mesh (N=20) → warm-started fine mesh via `solve_from_guess()` and `resample_solution()`

### State Vector (11D)

| Index | Symbol | Description | Group |
|-------|--------|-------------|-------|
| 0 | u | Body x-velocity | Plant |
| 1 | w | Body z-velocity | Plant |
| 2 | θ | Pitch angle | Plant |
| 3 | q | Pitch rate | Plant |
| 4 | h | Altitude | Plant |
| 5 | as | Moving mass accel | Mass |
| 6 | vs | Moving mass vel | Mass |
| 7 | xs | Moving mass pos | Mass |
| 8 | δt | Throttle | Aug. ctrl |
| 9 | δe | Elevator | Aug. ctrl |
| 10 | us | Moving mass input | Aug. ctrl |

### Collocation

Configurable Radau collocation degree (default d=1). For d=1 (backward Euler), the coefficient matrices are:

- `C = [[-1, -1], [1, 1]]` — derivative at collocation points
- `D = [0, 1]` — continuity (endpoint property of Radau)
- `B = [0.5, 0.5]` — quadrature weights

Higher degrees (d=2, d=3) use more internal collocation points per interval for better accuracy, but may require L-BFGS Hessian approximation for convergence.

### IPOPT Tuning

Configurable via `TrajectoryOptConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ipopt_mu_strategy` | `'adaptive'` | Barrier update: `'adaptive'` or `'monotone'` |
| `ipopt_hessian_approximation` | `'exact'` | `'exact'` or `'limited-memory'` (L-BFGS) |
| `ipopt_warm_start` | `False` | Enable warm-start from previous solution |

### Convergence Strategy

The optimizer uses several techniques to improve IPOPT convergence:

1. **S-curve initial guess**: Cubic Hermite interpolation for theta, parabolic pitch rate, clamped throttle ramp
2. **Relaxed us bounds**: `[-0.01, 0.01]` instead of `[0, 0]` to reduce tight-bound equality constraints
3. **Two-stage solve**: Coarse mesh (N=20, 500 iter) → fine mesh warm-started via `resample_solution()`

## Trim Analysis

The trim module (`trim.py`) finds equilibrium points across the flight envelope by solving force/moment balance equations. Refactored from `matlab/trim/`.

### Trim Problem Formulation

At each (V, θ) operating point, find control inputs (throttle, elevator) and angle of attack α such that:

```
Fx = L·sin(α) - D·cos(α) + 2·Fprop - m·g·sin(θ) = 0
Fz = -L·cos(α) - D·sin(α) + m·g·cos(θ) = 0
M  = M_aero - ms·g·xs = 0
```

Decision variables: `x = [throttle, ele_deg, alpha_rad]`

### Optimization Approach

- **Optimizer**: `scipy.optimize.minimize(method='SLSQP')` with bound constraints
- **Bounds**: throttle ∈ [0, 1], elevator ∈ [-30°, 20°], alpha ∈ [θ±45°]
- **Random restarts**: Up to 15 random initial points to avoid local minima (matches MATLAB `trim_Alpha.m`)

### Trim Modes

| Mode | Function | Description |
|------|----------|-------------|
| Level flight | `trim_level_flight()` | θ = α, γ = 0. Maps `trim_Theta.m` |
| General | `trim_at_conditions()` | Free α search. Maps `trim_Alpha.m` |
| Max moment | `trim_max_moment()` | Max residual M with force constraint. Maps `trim_MaxMy.m` |

### Transition Corridor

The corridor sweep (`sweep_corridor()`) evaluates trim feasibility over a (V, θ) grid. At each point:
1. Find equilibrium trim (`trim_at_conditions`)
2. Find max residual moment for forward/backward transition (`trim_max_moment`)
3. Flag is 1 if equilibrium is found and at least one transition moment exists

Uses `concurrent.futures.ProcessPoolExecutor` for parallel computation (replaces MATLAB `parfor`).

### Aerodynamic Model (Trim)

The trim module uses a simplified aero model (matching `trim_func.m`):
- Lift/Drag: from alpha lookup only (no elevator increment)
- Moment: from alpha lookup + elevator increment via slipstream dynamic pressure
- Propulsion: throttle → thrust lookup, slipstream velocity via momentum theory

## Linearization

The linearization module (`linearization.py`) computes Jacobians and analyzes system properties at trim points. Refactored from `matlab/trim/linearization/`.

### Linearized Dynamics

4-DOF state: `[V, α, θ, q]`, 2D control: `[throttle, elevator]`

```
V̇  = (-D - m·g·sin(θ-α) + 2·Fprop·cos(α)) / m
α̇  = (-L + m·g·cos(θ-α) - 2·Fprop·sin(α)) / (m·V) + q
θ̇  = q
q̇  = (M - ms·g·xs) / Jy
```

### Numerical Linearization

Jacobian matrices A (4×4) and B (4×2) are computed via central finite differences:

```
A[:,j] = (f(x+ε·ej) - f(x-ε·ej)) / (2ε)
B[:,j] = (f(u+ε·ej) - f(u-ε·ej)) / (2ε)
```

Default perturbation: ε = 10⁻⁶

### System Analysis

| Analysis | Function | Metrics |
|----------|----------|---------|
| Eigenvalue | `analyze_eigenvalues()` | Real/complex count, max/min real part |
| Controllability | `analyze_controllability()` | Rank, PBH test, Gramian SVD, modal controllability |
| Stability | `analyze_stability()` | Type, margin, damping ratios |
| Modal | `analyze_modes()` | Time constants, natural freq, overshoot |

The `sweep_linearization()` function runs analysis across the entire corridor grid, adding eigenvalue/controllability heatmaps to the TrimMesh.
