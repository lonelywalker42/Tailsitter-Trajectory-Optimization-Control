# Trajectory Optimization — Convergence Plan

## Current Status

The CasADi + IPOPT trajectory optimizer is structurally complete but does not converge.
IPOPT hits max iterations (2000) with constraint violation ~5e8 and dual infeasibility ~1.7e7.

### What Works

- Module structure: `trajectory_optimization.py`, `trajectory_plotting.py`, `trajectory_optimize.py`
- CasADi `nlpsol` interface with manual NLP construction
- Backward Euler (d=1 Radau) collocation — only degree that runs without algebraic errors
- CasADi `interpolant()` for cfg1 aero/propulsion tables
- Unit tests pass (config, interpolants, collocation coefficients, dynamics, build)

### Root Cause Analysis

| Problem | Impact | Evidence |
|---------|--------|----------|
| Poor initial guess | IPOPT starts in infeasible region | Linear interpolation doesn't satisfy dynamics |
| Tight us bounds [0,0] | 51 extra equality constraints | IPOPT counts lb==ub as equality |
| Non-smooth aero tables | Gradient noise disrupts search | 33-point linear interpolant |
| tf collapses to lower bound | Solver shortcuts transition time | tf=1.24s vs tf_guess=8s |

## Next Steps (Priority Order)

### 1. Improve Initial Guess

Replace linear interpolation with a physically motivated trajectory:

```python
# S-curve for theta: 4.9° → 90° over tf
theta_guess = x0[2] + (xf[2] - x0[2]) * (3*s**2 - 2*s**3)

# Smooth throttle ramp
dt_guess = x0[8] + (xf[8] - x0[8]) * s

# Small pitch rate during transition
q_guess = 6 * (xf[2] - x0[2]) / tf * s * (1 - s)  # parabola
```

Also: set `w0[0]` (Tf) to a value closer to the expected solution (e.g., 5–8s).

### 2. Relax us Bounds

Change `x_min[10]` and `x_max[10]` from `[0, 0]` to `[-0.01, 0.01]`.
This removes 51 tight-bound equality constraints, giving IPOPT more freedom.
The objective weight `w_dus=10000` will still drive us toward 0.

### 3. Two-Stage Solve

1. **Coarse**: N=20, max_iter=500 — fast, rough solution
2. **Fine**: N=50, max_iter=2000 — warm-started from coarse solution

Implementation: add `solve_from_guess(w0)` method that accepts an external initial guess.

### 4. IPOPT Tuning

If the above doesn't converge, try:

```python
opts = {
    'ipopt.mu_strategy': 'monotone',      # instead of 'adaptive'
    'ipopt.hessian_approximation': 'limited-memory',  # L-BFGS
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.warm_start_bound_push': 1e-6,
    'ipopt.warm_start_mult_bound_push': 1e-6,
}
```

### 5. Restore Aerodynamic Fidelity

Once convergence is achieved:

- Restore elevator sign: `dcm(-de)` (matching MATLAB `de=-u(2)`)
- Add CL/CD elevator increments: `cL + dcL(-de)`, `cD + dcD(-de)`
- These were removed because they made the problem harder to solve

### 6. Upgrade Collocation Degree

After d=1 converges, try d=2 or d=3 for better accuracy:

- d=2: 2 internal collocation points per interval
- d=3: 3 internal points (matching MATLAB GPOPS-II Radau)
- May need L-BFGS Hessian approximation for higher degrees

## Verification Checklist

- [ ] IPOPT returns `Solve_Succeeded` or `Solved_To_Acceptable_Level`
- [ ] Boundary conditions satisfied: `|x0 - x0_target| < 1e-3`, `|xf - xf_target| < 1e-3`
- [ ] Path constraint: `h ∈ [-50, 50]` for all nodes
- [ ] tf is reasonable (5–15s for forward↔hover transition)
- [ ] Trajectory plots match MATLAB results qualitatively
- [ ] All unit tests pass
- [ ] CLI: `python scripts/trajectory_optimize.py --direction forward2hover` runs end-to-end

## Files

| File | Purpose |
|------|---------|
| `src/tailsitter/trajectory_optimization.py` | Core optimizer module |
| `src/tailsitter/trajectory_plotting.py` | Visualization |
| `scripts/trajectory_optimize.py` | CLI entry point |
| `tests/test_trajectory_optimization.py` | Unit tests |
| `matlab/trajectory/LonDyn.m` | Reference: dynamics + aero tables |
| `matlab/trajectory/TraOptMain.m` | Reference: problem setup + bounds |
| `matlab/trajectory/UAVContinuous.m` | Reference: integrand + path constraints |
