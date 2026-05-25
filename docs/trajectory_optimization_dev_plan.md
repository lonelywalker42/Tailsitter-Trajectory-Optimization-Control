# Trajectory Optimization — Development Plan

## Goal

Make the CasADi + IPOPT trajectory optimizer converge for forward↔hover transitions. Current status: IPOPT hits max iterations (2000–5000) with constraint violation ~5e8 and dual infeasibility ~1.7e7.

## Changes Overview

Six sequential steps, each building on the previous. Commit after each step with verification.

---

### Step 1: Improve Initial Guess

**Why:** Linear interpolation between trim points doesn't satisfy dynamics, putting IPOPT in an infeasible region from the start.

**What to change in `src/tailsitter/trajectory_optimization.py`:**

Replace the linear interpolation block (lines ~487–505) with a physically motivated S-curve guess:

```python
for k in range(N + 1):
    s = k / N
    # S-curve (cubic Hermite) for theta: smooth pitch-over
    theta_k = x0_guess[2] + (xf_guess[2] - x0_guess[2]) * (3*s**2 - 2*s**3)
    # Pitch rate: parabola peaking at mid-transition
    q_k = 6 * (xf_guess[2] - x0_guess[2]) / tf_guess * s * (1 - s)
    # Throttle: linear ramp (already smooth)
    dt_k = x0_guess[8] + (xf_guess[8] - x0_guess[8]) * s
    # Other states: linear interpolation (u, w, h, as, vs, xs, de, us)
    xk = x0_guess + s * (xf_guess - x0_guess)
    xk[2] = theta_k
    xk[3] = q_k
    xk[8] = dt_k
    w0.extend(xk.tolist())
```

Also set `w0[0]` (Tf) to `tf_guess` explicitly (currently implicit from the loop).

**Verification:**
- Unit tests still pass
- IPOPT starts with lower initial constraint violation (check `inf_pr` in first iteration)

---

### Step 2: Relax `us` Bounds

**Why:** `x_min[10] = x_max[10] = 0` creates 51 (or N+1) tight-bound equality constraints that IPOPT counts explicitly, adding unnecessary constraint burden.

**What to change in `src/tailsitter/trajectory_optimization.py`:**

In `_build_bounds()` (or wherever `x_min`/`x_max` are defined), change:
```python
# Before:
x_min[10] = 0.0
x_max[10] = 0.0

# After:
x_min[10] = -0.01
x_max[10] = 0.01
```

The objective weight `w_dus = 10000` will still drive `us` toward 0. The solver gains freedom to explore without violating constraints.

**Verification:**
- Unit tests pass
- IPOPT reports fewer equality constraints in problem statistics

---

### Step 3: Two-Stage Solve (Coarse → Fine)

**Why:** A coarse mesh (N=20) solves fast and provides a warm start for the fine mesh (N=50+), dramatically improving convergence.

**What to change in `src/tailsitter/trajectory_optimization.py`:**

1. Add a `solve_from_guess(w0)` method that accepts an external initial guess vector and re-solves.
2. Add a `resample_solution(sol, N_new)` method that interpolates a solution onto a finer mesh.

**What to change in `scripts/trajectory_optimize.py`:**

Implement two-stage solve logic:
```python
# Stage 1: Coarse
coarse_cfg = TrajOptConfig(num_nodes=20, ipopt_max_iter=500)
coarse_opt = TrajectoryOptimizer(coarse_cfg)
coarse_result = coarse_opt.build_and_solve(direction)

# Stage 2: Fine (warm-started)
fine_cfg = TrajOptConfig(num_nodes=50, ipopt_max_iter=2000)
fine_opt = TrajectoryOptimizer(fine_cfg)
w0_fine = fine_opt.resample_solution(coarse_result, N_new=50)
fine_result = fine_opt.solve_from_guess(w0_fine)
```

**Verification:**
- Coarse solve completes in <30s
- Fine solve converges in fewer iterations than a cold start
- Boundary conditions satisfied on fine solution

---

### Step 4: IPOPT Tuning

**Why:** If Steps 1–3 don't fully converge, IPOPT algorithm options can help.

**What to change in `src/tailsitter/trajectory_optimization.py`:**

Add warm-start and strategy options to the IPOPT opts dict:
```python
opts = {
    'ipopt.max_iter': cfg.ipopt_max_iter,
    'ipopt.tol': cfg.ipopt_tol,
    'ipopt.print_level': cfg.ipopt_print_level,
    'ipopt.mu_strategy': 'monotone',           # changed from 'adaptive'
    'ipopt.hessian_approximation': 'limited-memory',  # L-BFGS
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.warm_start_bound_push': 1e-6,
    'ipopt.warm_start_mult_bound_push': 1e-6,
    'ipopt.linear_solver': 'mumps',
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.mu_init': 0.1,
}
```

Add these as configurable fields in `TrajOptConfig` with sensible defaults.

**Verification:**
- IPOPT converges to `Solve_Succeeded` or `Solved_To_Acceptable_Level`
- `tf` is in range [5, 15] seconds (not collapsed to lower bound)

---

### Step 5: Restore Aerodynamic Fidelity

**Why:** Elevator sign and CL/CD increments were removed to simplify the problem. Now that convergence is achieved, restore them.

**What to change in `src/tailsitter/trajectory_optimization.py` in `_ode()`:**

1. **Restore elevator sign:** Change `dcm(de)` to `dcm(-de)` to match MATLAB `de=-u(2)`.
2. **Add CL/CD elevator increments:** `cL + dcL(-de)`, `cD + dcD(-de)` using the aerosurface interpolants.

**Verification:**
- Optimizer still converges with restored aero
- Trajectory plots show physically reasonable alpha and elevator usage
- Qualitative comparison with MATLAB `ResultPlot.m` output

---

### Step 6: Upgrade Collocation Degree (Optional)

**Why:** d=1 (backward Euler) is first-order accurate. Higher degrees give better accuracy per node.

**What to change:**

Make `d` configurable in `TrajOptConfig` (default stays 1). After d=1 converges reliably:
- Try d=2 (2 internal collocation points per interval)
- Try d=3 (3 internal points, matching MATLAB GPOPS-II Radau)
- May need L-BFGS Hessian approximation for higher degrees (already added in Step 4)

**Verification:**
- d=2 and d=3 converge
- Solution accuracy improves (fewer nodes needed for same quality)
- Unit tests updated for new collocation degrees

---

## Verification Checklist (Final)

After all steps, verify:

- [ ] IPOPT returns `Solve_Succeeded` or `Solved_To_Acceptable_Level`
- [ ] Boundary conditions: `|x0 - x0_target| < 1e-3`, `|xf - xf_target| < 1e-3`
- [ ] Path constraint: `h ∈ [-50, 50]` for all nodes
- [ ] `tf` is reasonable (5–15s for forward↔hover transition)
- [ ] Trajectory plots match MATLAB results qualitatively
- [ ] All unit tests pass
- [ ] CLI: `python scripts/trajectory_optimize.py --direction forward2hover` runs end-to-end
- [ ] CHANGELOG.md updated

## Commit Plan

| Step | Commit Message |
|------|---------------|
| 1 | `fix: improve trajectory initial guess with S-curve interpolation` |
| 2 | `fix: relax us bounds to reduce IPOPT equality constraints` |
| 3 | `feat: add two-stage coarse-to-fine trajectory solve` |
| 4 | `tune: switch IPOPT to monotone barrier with L-BFGS Hessian` |
| 5 | `fix: restore elevator sign and CL/CD increments in trajectory dynamics` |
| 6 | `feat: make collocation degree configurable (d=1,2,3)` |

## Files Modified

| File | Steps |
|------|-------|
| `src/tailsitter/trajectory_optimization.py` | 1, 2, 3, 4, 5, 6 |
| `scripts/trajectory_optimize.py` | 3 |
| `tests/test_trajectory_optimization.py` | 3, 6 |
| `CHANGELOG.md` | All |
