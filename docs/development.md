# Development Guide

This document describes the development workflow, milestones, and contribution guidelines for the TailsitterControl project.

## Development Workflow

### Feature Development

1. **Plan**: Create a plan file describing the feature
2. **Implement**: Write code following the architecture
3. **Validate**: Run tests and verify functionality
4. **Commit**: Single focused commit with `feat:` prefix

### Refactoring

1. **Plan**: Describe what changes and why
2. **Implement**: Make structural changes without behavior changes
3. **Validate**: All existing tests must pass
4. **Commit**: Single commit with `refactor:` prefix

### Bug Fixes

1. **Reproduce**: Write a failing test that demonstrates the bug
2. **Fix**: Minimal change to make the test pass
3. **Validate**: All tests pass
4. **Commit**: Single commit with `fix:` prefix

## Commit Conventions

Use semantic commit messages:

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat: add reward shaping for altitude tracking` |
| `fix` | Bug fix | `fix: correct elevator sign inversion` |
| `refactor` | Code restructuring | `refactor: extract aero lookup method` |
| `docs` | Documentation | `docs: update user guide` |
| `perf` | Performance | `perf: vectorize coefficient lookup` |
| `test` | Tests | `test: add dynamics edge cases` |
| `chore` | Build/tooling | `chore: update dependencies` |

## Project Milestones

### Phase 1: Project Skeleton ✓

- [x] Create `pyproject.toml` with dependencies
- [x] Create YAML configuration files
- [x] Implement `config.py` with dataclasses and loaders

### Phase 2: Data Conversion ✓

- [x] Implement `convert_mat_to_npy.py`
- [x] Convert all `.mat` and `.xlsx` files to `.npy`/`.csv`

### Phase 3: Core Modules ✓

- [x] Implement `normalization.py`
- [x] Implement `reset.py`
- [x] Implement `reward.py`
- [x] Implement `dynamics.py`
- [x] Implement `env.py`

### Phase 4: Training Scripts ✓

- [x] Implement `train.py`
- [x] Implement `train_continue.py`

### Phase 5: Evaluation ✓

- [x] Implement `plotting.py`
- [x] Implement `evaluate.py`

### Phase 6: Testing ✓

- [x] Write unit tests for normalization
- [x] Write unit tests for reward
- [x] Write unit tests for environment
- [x] All tests passing (24/24)

### Phase 7: Trajectory Optimization (In Progress)

- [x] Implement `trajectory_optimization.py` — CasADi direct collocation (backward Euler)
- [x] Implement `trajectory_plotting.py` — visualization module
- [x] Implement `trajectory_optimize.py` — CLI entry point
- [x] Write unit tests for trajectory optimization
- [x] Improve initial guess (S-curve interpolation for theta/q/dt)
- [x] Relax us bounds to remove tight-bound equality constraints
- [x] Restore elevator sign fix (`dcm(-de)`) and CL/CD increments
- [x] Add two-stage coarse-to-fine solve (`solve_from_guess`, `resample_solution`)
- [x] Add IPOPT tuning options (mu_strategy, hessian_approximation, warm_start)
- [x] Make collocation degree configurable (d=1,2,3)
- [x] Update CHANGELOG.md
- [ ] **IPOPT convergence verification** — run full solve and verify convergence
- [ ] End-to-end verification (import → test → full solve → plot)

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_env.py -v

# Run specific test
python -m pytest tests/test_env.py::test_observation_space_shape -v

# Run with coverage
python -m pytest tests/ --cov=tailsitter
```

## Documentation Sync

When making changes, update documentation as follows:

| Change Type | Update Required |
|-------------|-----------------|
| `feat` | CHANGELOG.md, docs/architecture.md (if design changes), docs/user_guide.md (if user-facing) |
| `fix` | CHANGELOG.md |
| `refactor` | docs/architecture.md (if structure changes) |
| `docs` | Update the specific doc file only |
| `perf` | CHANGELOG.md |

## Adding New Features

### Adding a New Reward Component

1. Edit `src/tailsitter/reward.py`
2. Add the component to the `compute()` method
3. Update the reward formula in `docs/architecture.md`
4. Add tests in `tests/test_reward.py`
5. Commit with `feat: add [component] to reward function`

### Adding a New Termination Condition

1. Edit `src/tailsitter/env.py` in `_check_terminated()`
2. Update the termination table in `docs/architecture.md`
3. Add tests in `tests/test_env.py`
4. Commit with `feat: add [condition] termination`

### Modifying Dynamics

1. Edit `src/tailsitter/dynamics.py`
2. Verify against original MATLAB code in `mdl/`
3. Run dynamics tests
4. Update `docs/architecture.md` if equations change
5. Commit with `refactor: modify [component] dynamics`

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and small
- Use docstrings for public methods
- Prefer explicit over implicit

## Dependencies

Core dependencies (see `pyproject.toml`):
- `gymnasium>=0.29` — RL environment interface
- `stable-baselines3>=2.3` — RL algorithms (SAC, PPO)
- `torch>=2.0` — Neural network backend
- `numpy` — Numerical computing
- `scipy` — Scientific computing (interpolation, .mat loading)
- `pyyaml` — Configuration files
- `matplotlib` — Visualization
- `pandas` — Data handling

Dev dependencies:
- `pytest` — Testing framework
- `pytest-cov` — Test coverage
