# Trajectory Optimization — Convergence Plan

## Current Status

Steps 1–6 of the convergence plan have been implemented. The optimizer now has:
- S-curve initial guess for theta/q/dt
- Relaxed `us` bounds `[-0.01, 0.01]`
- Restored elevator sign (`dcm(-de)`) and CL/CD increments
- Two-stage coarse-to-fine solve with warm-starting
- Configurable IPOPT tuning (mu_strategy, hessian_approximation, warm_start)
- Configurable collocation degree (d=1,2,3)

**Remaining**: Run a full end-to-end solve to verify IPOPT convergence and validate trajectory quality against MATLAB results.

## Completed Steps

### 1. Improve Initial Guess ✓

Replaced linear interpolation with S-curve (cubic Hermite) for theta, parabolic pitch rate, and clamped throttle ramp.

### 2. Relax us Bounds ✓

Changed `x_min[10]` and `x_max[10]` from `[0, 0]` to `[-0.01, 0.01]`. The objective weight `w_dus=10000` still drives us toward 0.

### 3. Two-Stage Solve ✓

Added `solve_from_guess(w0)` and `resample_solution(result, N_new)` methods. CLI supports `--two-stage` flag.

### 4. IPOPT Tuning ✓

Added configurable `ipopt_mu_strategy`, `ipopt_hessian_approximation`, and `ipopt_warm_start` to `TrajectoryOptConfig`. CLI supports `--mu-strategy`, `--hessian` arguments.

### 5. Restore Aerodynamic Fidelity ✓

Restored elevator sign inversion (`dcm(-de)`) and added CL/CD elevator increments (`dcL(-de)`, `dcD(-de)`) matching MATLAB `LonDyn.m`.

### 6. Upgrade Collocation Degree ✓

Made `collocation_degree` configurable (d=1,2,3). Default remains d=1. CLI supports `--collocation-degree` argument.

## Remaining Verification

- [ ] IPOPT returns `Solve_Succeeded` or `Solved_To_Acceptable_Level` with full problem (N=100+)
- [ ] Boundary conditions satisfied: `|x0 - x0_target| < 1e-3`, `|xf - xf_target| < 1e-3`
- [ ] Path constraint: `h ∈ [-50, 50]` for all nodes
- [ ] tf is reasonable (5–15s for forward↔hover transition)
- [ ] Trajectory plots match MATLAB results qualitatively
- [ ] All unit tests pass (currently 25/25 for trajectory, 67 total)

## Files

| File | Purpose |
|------|---------|
| `src/tailsitter/trajectory_optimization.py` | Core optimizer module |
| `src/tailsitter/trajectory_plotting.py` | Visualization |
| `scripts/trajectory_optimize.py` | CLI entry point |
| `tests/test_trajectory_optimization.py` | Unit tests (25 tests) |
| `matlab/trajectory/LonDyn.m` | Reference: dynamics + aero tables |
| `matlab/trajectory/TraOptMain.m` | Reference: problem setup + bounds |
| `matlab/trajectory/UAVContinuous.m` | Reference: integrand + path constraints |
