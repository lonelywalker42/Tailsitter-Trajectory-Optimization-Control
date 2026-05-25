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
