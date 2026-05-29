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
- **Collocation**: Configurable Radau degree (d=1 backward Euler, d=2, d=3); default d=2 (Radau) for better accuracy
- **Dynamics**: 11D state (8 plant + 3 augmented controls), 3D control (rates), matching MATLAB `LonDyn.m` cfg1 tables with induced velocity model (momentum theory)
- **Aero model**: CL/CD with elevator increments (`dcL(-de)`, `dcD(-de)`), elevator sign inversion matching MATLAB
- **Objective**: `min(w_tf·tf + ∫ L(x,u) dt)` with normalized weights (O(1))
- **Boundary conditions**: Penalty-based (`w_bc * sumsqr`) with continuation (100→1e6) to avoid NLP infeasibility
- **Auto-trim**: Boundary conditions computed from optimizer dynamics via `_find_trim()` grid search
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

### Induced Velocity Model

The optimizer dynamics include an induced velocity model based on momentum theory:

```
v_ind = sqrt(Fprop / (2·ρ·π·R²))
q_ind = 0.5·ρ·(v_ind + V·cos(α))²
```

- Base aerodynamic forces (L, D) use freestream dynamic pressure `q0 = 0.5·ρ·V²`
- Elevator moment uses slipstream dynamic pressure `q_ind` (higher than q0 at low speed)
- At V=20 m/s: `q_ind/q0 ≈ 1.24` (24% increase in effective dynamic pressure)

This model significantly changes the force balance compared to the RL environment's simpler dynamics (cfg2), particularly at low speeds where induced velocity dominates.

### Trim Solver

The `_find_trim()` method finds equilibrium points within the optimizer dynamics:

1. Decision variables: `[α, θ, δt, δe]`
2. Constraints: `udot=0, wdot=0, qdot=0` with `V=sqrt(u²+w²)=V_target`
3. Optional penalties for flight path angle (γ=θ-α) or pitch angle (θ)
4. Grid search over initial guesses to avoid local minima

**Key finding**: With induced velocity dynamics, V=20 m/s has no level-flight (γ=0) trim. The only equilibria are climbing flight trims (e.g., γ=60°, θ=58°).

### Collocation

Configurable Radau collocation degree (default d=2). For d=2, each interval has 2 interior collocation states (Z), adding 2·N·11 decision variables. Coefficient matrices are computed via `cs.collocation_points(d, 'radau')`.

For d=1 (backward Euler), the coefficient matrices are:

- `C = [[-1, -1], [1, 1]]` — derivative at collocation points
- `D = [0, 1]` — continuity (endpoint property of Radau)
- `B = [0.5, 0.5]` — quadrature weights

Defect constraints use velocity-error form scaled by 1/h: `(X[k+1] - X[k])/h - f(X[k+1]) = 0` for better NLP conditioning.

### IPOPT Tuning

Configurable via `TrajectoryOptConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ipopt_mu_strategy` | `'adaptive'` | Barrier update: `'adaptive'` or `'monotone'` |
| `ipopt_hessian_approximation` | `'exact'` | `'exact'` or `'limited-memory'` (L-BFGS) |
| `ipopt_warm_start` | `False` | Enable warm-start from previous solution |
| `ipopt_nlp_scaling_max_gradient` | `100` | Max gradient for NLP scaling |
| `mumps_pivtol` | `1e-4` | MUMPS pivoting tolerance for near-singular KKT |
| `acceptable_tol` | `1e-3` | Acceptable convergence tolerance (early termination) |

### Boundary Conditions

Boundary conditions are enforced via penalty terms in the objective rather than hard equality constraints:

```
J += w_bc * ‖X[0] - x0_target‖²
J += w_bc * ‖X[N] - xf_target‖²
```

The `solve()` method uses continuation on `w_bc` (100→1000→10000→100000→1e6), warm-starting each step from the previous solution. This avoids the NLP infeasibility that occurs with hard equality constraints when the initial guess is far from feasibility.

Boundary target variables (`x0_target`, `xf_target`) are fixed decision variables (lb=ub=value) in the NLP, ensuring they remain constant across all solves.

### Path Constraints

| Constraint | Range | Description |
|------------|-------|-------------|
| Altitude h | [-20, 50] m | Prevents ground crash and excessive altitude |
| Alpha α | [-60°, 120°] | Prevents extreme angles of attack |
| Theta θ | [-π, 120°] | Allows full transition corridor |

### Convergence Strategy

The optimizer uses several techniques to improve IPOPT convergence:

1. **Auto-trim boundary conditions**: `_find_trim()` computes equilibrium points in the optimizer dynamics, avoiding model mismatch between boundary conditions and dynamics
2. **Penalty-based BCs with continuation**: Progressive enforcement avoids infeasibility from poor initial guesses
3. **S-curve initial guess**: Cubic Hermite interpolation for theta, parabolic pitch rate, clamped throttle ramp
4. **Velocity-error collocation**: Defect constraints scaled by 1/h measure velocity error (~O(1)) instead of position error (~O(h))
5. **Normalized weights**: All objective weights are O(1) for better gradient scaling
6. **Two-stage solve**: Coarse mesh (N=20, 500 iter) → fine mesh warm-started via `resample_solution()`

### Known Limitations

The induced velocity dynamics model creates stiff, nonlinear regions that make trajectory optimization very difficult:

- IPOPT often converges to "Optimal Solution Found" but with boundary conditions partially unsatisfied
- The optimizer may get stuck at intermediate flight states (e.g., V≈3.6 m/s, θ≈-6°) instead of reaching the hover target
- The forward trim (V=20) has no level-flight equilibrium — only climbing flight (γ≈60°), making the transition to hover inherently difficult
- Forward simulation from hover trim with constant controls can blow up to NaN, indicating stiff dynamics

These issues are fundamental to the induced velocity dynamics model, not solver configuration problems.

## Differential Flatness Trajectory Optimization

The differential flatness (DF) module provides an alternative trajectory optimization approach using the flatness property of the longitudinal dynamics. Given flat outputs (V, γ) and their derivatives, the full state and control can be recovered via algebraic computation.

### Design

- **Approach**: Exploits differential flatness — the longitudinal dynamics are differentially flat with flat outputs (V, γ)
- **Optimizer**: `scipy.optimize.minimize(method='trust-constr')` with nonlinear constraints
- **Decision variables**: `[dt_1..dt_N, dγ_1..dγ_N]` — time increments and FPA increments along a velocity grid
- **Objective**: Weighted sum of time, height change, control energy, pitch rate safety margin, and smoothness
- **Constraints**: Gamma rate bounds, alpha bounds, pitch rate limits, throttle non-negativity, terminal gamma

### Architecture

| Module | File | Description |
|--------|------|-------------|
| DF Transfer | `differential_flatness.py` | Newton iteration on α, force/moment solve, elevator inverse lookup |
| Sin4 Model | `differential_flatness.py` | `∑ aᵢ·sin(bᵢ·α + cᵢ)` with analytical 1st/2nd derivatives |
| Pitch Rate Margin | `differential_flatness.py` | Max feasible pitch rate from control margin and effectiveness |
| DF Corridor | `df_trim.py` | (V, γ) grid sweep with feasibility flags |
| DF Optimizer | `df_trajectory_optimization.py` | (dt, dγ) optimization along velocity grid |
| Baseline Path | `baseline_trajectory.py` | Geometric path optimization maximizing boundary distance |
| DF Plotting | `df_trajectory_plotting.py` | Trajectory, corridor, comparison visualization |

### DF Transfer Algorithm

The core algorithm (`differential_flatness_transfer()`) solves for the full state given flat outputs:

1. **Newton iteration on α**: Solves `f(α) = m·(V̇·sinα - V·γ̇·cosα + g·sinγ·sinα - g·cosγ·cosα) + q_dyn·cz(α) = 0`
2. **Theta**: `θ = γ + α`
3. **Thrust**: From force equilibrium (special case at α=90°)
4. **Pitch rate q, Ṫ**: From 2×2 linear system (force balance derivatives)
5. **q̈, T̈**: From another 2×2 linear system (second-order balance)
6. **Elevator**: From external moment via induced velocity model + inverse lookup `interp(dcm, ele, dcm_d)`

Convergence: tolerance=2e-3, max_iter=50, initial guess α=0.

### Sinusoidal Aero Model (sin4)

The sin4 model provides analytical derivatives needed for Newton iteration:

```
f(α_deg) = ∑ᵢ₌₁⁴ aᵢ·sin(bᵢ·α_deg + cᵢ)
```

Each coefficient array is 12-dimensional: `[a₁,b₁,c₁, a₂,b₂,c₂, a₃,b₃,c₃, a₄,b₄,c₄]`. Derivatives w.r.t. α in radians include a `*57.3` chain rule factor.

### DF Trajectory Optimization

The DF optimizer (`df_trajectory_optimize()`) finds optimal transitions in (V, γ) space:

- **Velocity grid**: Evenly spaced from V_start to V_end with step `dv` (default 1 m/s)
- **Decision variables**: `x = [dt₁..dt_N, dγ₁..dγ_N]` (2N variables)
- **Objective**: `J = w_time·T + w_height·‖h‖² + w_energy·‖T‖² - w_safety·‖q-q_max‖² + λ·smoothness`
- **Smoothness**: Angle between consecutive path segments in normalized (V/16, γ/90) space
- **Constraints**: γ̇ bounds, α bounds, |q| ≤ q_max, throttle ≥ 0, terminal γ

### Baseline Path Optimization

The baseline method (`optimize_baseline_path()`) finds a geometric corridor path:

1. Extract corridor boundary via contour at 0.5 level
2. Initialize control points along straight line, adjusting to feasible region
3. Optimize internal control points to maximize distance from boundary
4. Interpolate trim data (θ, α, T, δe, q_max) along optimal path

### Pitch Rate Margin

The max feasible pitch rate is computed from:

```
q_max = τ · (M_δe · η) / Jy
```

Where:
- `τ`: Correction time constant (clamped to [τ_min, τ_max])
- `M_δe`: Elevator control effectiveness (from induced velocity model)
- `η`: Control margin (min distance to elevator limits)

### Comparison: CasADi vs DF

| Aspect | CasADi (trajectory_optimization.py) | DF (df_trajectory_optimization.py) |
|--------|-------------------------------------|-------------------------------------|
| State space | 11D (full dynamics) | 2D flat outputs (V, γ) |
| Solver | IPOPT (interior point) | scipy trust-constr |
| Collocation | Radau (d=1,2,3) | Direct transcription |
| Dynamics | Full 5-DOF with induced velocity | Differential flatness transfer |
| Convergence | Sensitive to initial guess | More robust (lower dimensional) |
| Speed | Slower (large NLP) | Faster (small NLP) |

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
