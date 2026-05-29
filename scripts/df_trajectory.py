"""DF-based trajectory optimization script for tailsitter UAV transitions.

Replaces matlab/analysis/controller/trajectory_compare.m, Main.m, df_trim.m,
sim_main.m, sim_main_baseline.m, sim_compare.m, transition_sim_compare.m.

Usage:
    python scripts/df_trajectory.py df-trim
    python scripts/df_trajectory.py df-optimize --direction hover2forward
    python scripts/df_trajectory.py df-baseline
    python scripts/df_trajectory.py df-compare --direction hover2forward
    python scripts/df_trajectory.py df-simulate --direction hover2forward
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tailsitter.differential_flatness import load_sin4_coeffs, PitchRateMarginParams
from tailsitter.df_trim import df_trim_sweep, save_df_trim_mesh, load_df_trim_mesh
from tailsitter.df_trajectory_optimization import (
    df_trajectory_optimize, save_df_trajectory_result, load_df_trajectory_result,
    TrajectoryOptParams,
)
from tailsitter.baseline_trajectory import (
    optimize_baseline_path, save_baseline_result, load_baseline_result,
)
from tailsitter.df_trajectory_plotting import (
    plot_df_trajectory, plot_df_corridor_suite, plot_df_trajectory_comparison,
    plot_tracking_error,
)
from tailsitter.df_trajectory_simulation import (
    simulate_df_trajectory, simulate_baseline_trajectory,
    analyze_tracking_error, print_error_report,
)
from tailsitter.trim import load_aero_data as load_trim_aero_data
from tailsitter.dynamics import LongitudinalDynamics
from tailsitter.config import PhysicalConfig


def cmd_df_trim(args):
    """Run DF-based corridor sweep."""
    print("=" * 50)
    print("DF Trim Corridor Sweep")
    print("=" * 50)

    t0 = time.time()
    mesh = df_trim_sweep(
        V_range=(args.v_min, args.v_max),
        gamma_range=(args.gamma_min, args.gamma_max),
        dV=args.dv,
        dgamma=args.dgamma,
        Vdot=args.vdot,
        gammadot=args.gammadot,
        config_name=args.config,
        data_dir=args.data_dir,
        limit=(args.ele_min, args.ele_max),
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f} s")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"df_trim_{args.config}.npz"
    save_df_trim_mesh(mesh, npz_path)
    print(f"Results saved to {npz_path}")

    # Plot
    saved_plots = plot_df_corridor_suite(mesh, output_dir)
    for p in saved_plots:
        print(f"Plot saved to {p}")


def cmd_df_optimize(args):
    """Run DF-based trajectory optimization."""
    print("=" * 50)
    print("DF Trajectory Optimization")
    print("=" * 50)

    # Parse direction
    if args.direction == "hover2forward":
        vel = (0.1, 16.0)
        gamma_deg = (90.0, 0.0)
    elif args.direction == "forward2hover":
        vel = (16.0, 0.1)
        gamma_deg = (0.0, 90.0)
    else:
        vel = tuple(args.vel)
        gamma_deg = tuple(args.gamma)

    params = TrajectoryOptParams(
        w_time=args.w_time,
        w_height=args.w_height,
        w_energy=args.w_energy,
        w_safety=args.w_safety,
        dv=args.dv,
    )

    margin_params = PitchRateMarginParams(k=args.margin_k)

    print(f"Direction: {args.direction}")
    print(f"V: {vel[0]:.1f} -> {vel[1]:.1f} m/s")
    print(f"Gamma: {gamma_deg[0]:.1f} -> {gamma_deg[1]:.1f} deg")
    print()

    result = df_trajectory_optimize(
        vel=vel,
        gamma_deg=gamma_deg,
        params=params,
        config_name=args.config,
        data_dir=args.data_dir,
        margin_params=margin_params,
        verbose=True,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"df_traj_{args.direction}.npz"
    save_df_trajectory_result(result, npz_path)
    print(f"\nResults saved to {npz_path}")

    # Plot
    saved_plots = plot_df_trajectory(result, output_dir, filename=f"df_traj_{args.direction}")
    for p in saved_plots:
        print(f"Plot saved to {p}")


def cmd_df_baseline(args):
    """Run baseline corridor path optimization."""
    print("=" * 50)
    print("Baseline Corridor Path Optimization")
    print("=" * 50)

    # First run DF trim sweep if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / f"df_trim_{args.config}.npz"
    if mesh_path.exists() and not args.recompute:
        print(f"Loading existing trim mesh from {mesh_path}")
        mesh = load_df_trim_mesh(mesh_path)
    else:
        print("Running DF trim sweep...")
        mesh = df_trim_sweep(
            config_name=args.config,
            data_dir=args.data_dir,
            verbose=True,
        )
        save_df_trim_mesh(mesh, mesh_path)
        print(f"Trim mesh saved to {mesh_path}")

    # Parse start/end points
    start = (args.start_v, args.start_gamma)
    end = (args.end_v, args.end_gamma)

    print(f"\nOptimizing baseline path: ({start[0]:.1f}, {start[1]:.1f}) -> ({end[0]:.1f}, {end[1]:.1f})")

    result = optimize_baseline_path(
        df_trim_mesh=mesh,
        start_point=start,
        end_point=end,
        n_control_points=args.n_points,
        verbose=True,
    )

    # Save
    npz_path = output_dir / f"baseline_path_{args.direction}.npz"
    save_baseline_result(result, npz_path)
    print(f"\nResults saved to {npz_path}")


def cmd_df_compare(args):
    """Run both proposed and baseline trajectories, generate comparison."""
    print("=" * 50)
    print("DF Trajectory Comparison")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse direction
    if args.direction == "hover2forward":
        vel = (0.1, 16.0)
        gamma_deg = (90.0, 0.0)
        start = (0.1, 90.0)
        end = (16.0, 0.0)
    elif args.direction == "forward2hover":
        vel = (16.0, 0.1)
        gamma_deg = (0.0, 90.0)
        start = (16.0, 0.0)
        end = (0.1, 90.0)
    else:
        vel = tuple(args.vel)
        gamma_deg = tuple(args.gamma)
        start = (vel[0], gamma_deg[0])
        end = (vel[1], gamma_deg[1])

    # Step 1: DF trim sweep
    mesh_path = output_dir / f"df_trim_{args.config}.npz"
    if mesh_path.exists() and not args.recompute:
        print(f"Loading existing trim mesh from {mesh_path}")
        mesh = load_df_trim_mesh(mesh_path)
    else:
        print("Running DF trim sweep...")
        mesh = df_trim_sweep(
            config_name=args.config,
            data_dir=args.data_dir,
            verbose=True,
        )
        save_df_trim_mesh(mesh, mesh_path)

    # Step 2: DF trajectory optimization (proposed)
    print("\n--- Proposed trajectory (DF optimization) ---")
    proposed = df_trajectory_optimize(
        vel=vel,
        gamma_deg=gamma_deg,
        config_name=args.config,
        data_dir=args.data_dir,
        verbose=True,
    )
    save_df_trajectory_result(proposed, output_dir / f"df_traj_{args.direction}.npz")

    # Step 3: Baseline path optimization
    print("\n--- Baseline trajectory (corridor path) ---")
    baseline = optimize_baseline_path(
        df_trim_mesh=mesh,
        start_point=start,
        end_point=end,
        n_control_points=args.n_points,
        verbose=True,
    )
    save_baseline_result(baseline, output_dir / f"baseline_path_{args.direction}.npz")

    # Step 4: Comparison plots
    print("\n--- Generating comparison plots ---")
    saved_plots = plot_df_trajectory(proposed, output_dir, filename=f"df_traj_{args.direction}")
    for p in saved_plots:
        print(f"Plot saved to {p}")

    saved_plots = plot_df_trajectory_comparison(
        proposed, baseline, mesh, output_dir,
        filename=f"comparison_{args.direction}",
    )
    for p in saved_plots:
        print(f"Plot saved to {p}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Proposed: cost={proposed.cost:.6f}, time={proposed.time[-1]:.2f}s, "
          f"success={proposed.success}")
    print(f"Baseline: {len(baseline.optimal_path)} path points")
    print(f"{'='*50}")


def _load_aero_dict(config_name: str, data_dir: str) -> dict:
    """Load aero data as dict for LongitudinalDynamics."""
    aero = load_trim_aero_data(config_name, data_dir)
    from dataclasses import asdict
    return asdict(aero)


def _load_dynamics(config_name: str, data_dir: str) -> LongitudinalDynamics:
    """Initialize dynamics model from config."""
    phys = PhysicalConfig()
    aero_dict = _load_aero_dict(config_name, data_dir)
    return LongitudinalDynamics(phys, aero_dict)


def cmd_df_simulate(args):
    """Simulate trajectory tracking and analyze errors."""
    print("=" * 50)
    print("DF Trajectory Tracking Simulation")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse direction
    if args.direction == "hover2forward":
        vel = (0.1, 16.0)
        gamma_deg = (90.0, 0.0)
        start = (0.1, 90.0)
        end = (16.0, 0.0)
    elif args.direction == "forward2hover":
        vel = (16.0, 0.1)
        gamma_deg = (0.0, 90.0)
        start = (16.0, 0.0)
        end = (0.1, 90.0)
    else:
        vel = tuple(args.vel)
        gamma_deg = tuple(args.gamma)
        start = (vel[0], gamma_deg[0])
        end = (vel[1], gamma_deg[1])

    # Load dynamics model
    print(f"\nLoading dynamics model (config={args.config})...")
    dyn = _load_dynamics(args.config, args.data_dir)
    aero_dict = _load_aero_dict(args.config, args.data_dir)

    # Step 1: DF trajectory optimization (proposed)
    print("\n--- Proposed trajectory (DF optimization) ---")
    proposed = df_trajectory_optimize(
        vel=vel,
        gamma_deg=gamma_deg,
        config_name=args.config,
        data_dir=args.data_dir,
        verbose=True,
    )
    save_df_trajectory_result(proposed, output_dir / f"df_traj_{args.direction}.npz")

    # Step 2: Baseline path optimization
    mesh_path = output_dir / f"df_trim_{args.config}.npz"
    if mesh_path.exists():
        print(f"\nLoading existing trim mesh from {mesh_path}")
        mesh = load_df_trim_mesh(mesh_path)
    else:
        print("\nRunning DF trim sweep...")
        mesh = df_trim_sweep(
            config_name=args.config,
            data_dir=args.data_dir,
            verbose=True,
        )
        save_df_trim_mesh(mesh, mesh_path)

    print("\n--- Baseline trajectory (corridor path) ---")
    baseline = optimize_baseline_path(
        df_trim_mesh=mesh,
        start_point=start,
        end_point=end,
        n_control_points=args.n_points,
        verbose=True,
    )
    save_baseline_result(baseline, output_dir / f"baseline_path_{args.direction}.npz")

    # Step 3: Simulate proposed trajectory
    print("\n--- Simulating proposed trajectory ---")
    sim_proposed = simulate_df_trajectory(
        proposed, dyn, aero_dict, dt=args.dt, h0=args.h0,
    )
    error_proposed = analyze_tracking_error(sim_proposed)
    print_error_report(error_proposed, "Proposed")

    # Step 4: Simulate baseline trajectory
    print("\n--- Simulating baseline trajectory ---")
    sim_baseline = simulate_baseline_trajectory(
        baseline, dyn, aero_dict, dt=args.dt, acc=args.acc, h0=args.h0,
    )
    error_baseline = analyze_tracking_error(sim_baseline)
    print_error_report(error_baseline, "Baseline")

    # Step 5: Generate error plots
    print("\n--- Generating error analysis plots ---")
    saved_plots = plot_tracking_error(sim_proposed, error_proposed, output_dir, prefix="proposed_")
    for p in saved_plots:
        print(f"Plot saved to {p}")

    saved_plots = plot_tracking_error(sim_baseline, error_baseline, output_dir, prefix="baseline_")
    for p in saved_plots:
        print(f"Plot saved to {p}")

    # Step 6: Save simulation data
    np.savez(
        output_dir / f"sim_{args.direction}.npz",
        # Proposed
        proposed_time=sim_proposed.time,
        proposed_actual_V=sim_proposed.actual_V,
        proposed_actual_gamma=sim_proposed.actual_gamma,
        proposed_actual_theta=sim_proposed.actual_theta,
        proposed_actual_q=sim_proposed.actual_q,
        proposed_actual_alpha=sim_proposed.actual_alpha,
        proposed_actual_h=sim_proposed.actual_h,
        proposed_ref_V=sim_proposed.ref_V,
        proposed_ref_gamma=sim_proposed.ref_gamma,
        proposed_ref_theta=sim_proposed.ref_theta,
        proposed_ref_q=sim_proposed.ref_q,
        proposed_ref_alpha=sim_proposed.ref_alpha,
        proposed_ref_h=sim_proposed.ref_h,
        # Baseline
        baseline_time=sim_baseline.time,
        baseline_actual_V=sim_baseline.actual_V,
        baseline_actual_gamma=sim_baseline.actual_gamma,
        baseline_actual_theta=sim_baseline.actual_theta,
        baseline_actual_q=sim_baseline.actual_q,
        baseline_actual_alpha=sim_baseline.actual_alpha,
        baseline_actual_h=sim_baseline.actual_h,
        baseline_ref_V=sim_baseline.ref_V,
        baseline_ref_gamma=sim_baseline.ref_gamma,
        baseline_ref_theta=sim_baseline.ref_theta,
        baseline_ref_q=sim_baseline.ref_q,
        baseline_ref_alpha=sim_baseline.ref_alpha,
        baseline_ref_h=sim_baseline.ref_h,
    )
    print(f"\nSimulation data saved to {output_dir / f'sim_{args.direction}.npz'}")

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"{'Channel':<15} {'Proposed RMSE':>15} {'Baseline RMSE':>15}")
    print("-" * 45)
    print(f"{'V [m/s]':<15} {error_proposed.rmse_V:>15.4f} {error_baseline.rmse_V:>15.4f}")
    print(f"{'gamma [deg]':<15} {error_proposed.rmse_gamma:>15.4f} {error_baseline.rmse_gamma:>15.4f}")
    print(f"{'theta [deg]':<15} {error_proposed.rmse_theta:>15.4f} {error_baseline.rmse_theta:>15.4f}")
    print(f"{'q [deg/s]':<15} {error_proposed.rmse_q:>15.4f} {error_baseline.rmse_q:>15.4f}")
    print(f"{'alpha [deg]':<15} {error_proposed.rmse_alpha:>15.4f} {error_baseline.rmse_alpha:>15.4f}")
    print(f"{'h [m]':<15} {error_proposed.rmse_height:>15.4f} {error_baseline.rmse_height:>15.4f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="DF-based trajectory optimization for tailsitter UAV")
    parser.add_argument(
        "--config", type=str, default="aero_cfg2",
        help="Aero configuration name (default: aero_cfg2)")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Path to processed data directory (default: data/processed)")
    parser.add_argument(
        "--output-dir", type=str, default="results/df_trajectory",
        help="Output directory (default: results/df_trajectory)")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # df-trim subcommand
    p_trim = subparsers.add_parser("df-trim", help="Run DF corridor sweep")
    p_trim.add_argument("--v-min", type=float, default=0.0)
    p_trim.add_argument("--v-max", type=float, default=20.0)
    p_trim.add_argument("--dv", type=float, default=0.5, help="Velocity step [m/s]")
    p_trim.add_argument("--gamma-min", type=float, default=0.0)
    p_trim.add_argument("--gamma-max", type=float, default=90.0)
    p_trim.add_argument("--dgamma", type=float, default=1.0, help="FPA step [deg]")
    p_trim.add_argument("--vdot", type=float, default=0.0, help="Vdot [m/s^2]")
    p_trim.add_argument("--gammadot", type=float, default=0.0, help="gammadot [rad/s]")
    p_trim.add_argument("--ele-min", type=float, default=-20.0)
    p_trim.add_argument("--ele-max", type=float, default=30.0)

    # df-optimize subcommand
    p_opt = subparsers.add_parser("df-optimize", help="Run DF trajectory optimization")
    p_opt.add_argument("--direction", type=str, default="hover2forward",
                       choices=["hover2forward", "forward2hover", "custom"],
                       help="Transition direction")
    p_opt.add_argument("--vel", type=float, nargs=2, default=[0.1, 16.0],
                       metavar=("V_START", "V_END"),
                       help="Velocity range [m/s] (for custom direction)")
    p_opt.add_argument("--gamma", type=float, nargs=2, default=[90.0, 0.0],
                       metavar=("GAMMA_START", "GAMMA_END"),
                       help="FPA range [deg] (for custom direction)")
    p_opt.add_argument("--dv", type=float, default=1.0, help="Velocity grid step [m/s]")
    p_opt.add_argument("--w-time", type=float, default=0.1, help="Time weight")
    p_opt.add_argument("--w-height", type=float, default=0.0, help="Height weight")
    p_opt.add_argument("--w-energy", type=float, default=0.1, help="Energy weight")
    p_opt.add_argument("--w-safety", type=float, default=0.1, help="Safety weight")
    p_opt.add_argument("--margin-k", type=float, default=2.0, help="Pitch rate margin k")

    # df-baseline subcommand
    p_base = subparsers.add_parser("df-baseline", help="Run baseline corridor path optimization")
    p_base.add_argument("--direction", type=str, default="hover2forward",
                        choices=["hover2forward", "forward2hover", "custom"],
                        help="Transition direction")
    p_base.add_argument("--start-v", type=float, default=0.1, help="Start V [m/s]")
    p_base.add_argument("--start-gamma", type=float, default=90.0, help="Start gamma [deg]")
    p_base.add_argument("--end-v", type=float, default=16.0, help="End V [m/s]")
    p_base.add_argument("--end-gamma", type=float, default=0.0, help="End gamma [deg]")
    p_base.add_argument("--n-points", type=int, default=16, help="Number of control points")
    p_base.add_argument("--recompute", action="store_true",
                        help="Force recomputation of trim mesh")

    # df-compare subcommand
    p_cmp = subparsers.add_parser("df-compare", help="Run comparison: proposed vs baseline")
    p_cmp.add_argument("--direction", type=str, default="hover2forward",
                       choices=["hover2forward", "forward2hover", "custom"],
                       help="Transition direction")
    p_cmp.add_argument("--vel", type=float, nargs=2, default=[0.1, 16.0],
                       metavar=("V_START", "V_END"))
    p_cmp.add_argument("--gamma", type=float, nargs=2, default=[90.0, 0.0],
                       metavar=("GAMMA_START", "GAMMA_END"))
    p_cmp.add_argument("--n-points", type=int, default=16,
                       help="Number of baseline control points")
    p_cmp.add_argument("--recompute", action="store_true",
                        help="Force recomputation of trim mesh")

    # df-simulate subcommand
    p_sim = subparsers.add_parser("df-simulate",
                                   help="Simulate trajectory tracking and analyze errors")
    p_sim.add_argument("--direction", type=str, default="hover2forward",
                       choices=["hover2forward", "forward2hover", "custom"],
                       help="Transition direction")
    p_sim.add_argument("--vel", type=float, nargs=2, default=[0.1, 16.0],
                       metavar=("V_START", "V_END"),
                       help="Velocity range [m/s] (for custom direction)")
    p_sim.add_argument("--gamma", type=float, nargs=2, default=[90.0, 0.0],
                       metavar=("GAMMA_START", "GAMMA_END"),
                       help="FPA range [deg] (for custom direction)")
    p_sim.add_argument("--dt", type=float, default=0.02,
                       help="Simulation timestep [s] (default: 0.02)")
    p_sim.add_argument("--h0", type=float, default=0.0,
                       help="Initial altitude [m] (default: 0.0)")
    p_sim.add_argument("--acc", type=float, default=5.0,
                       help="Baseline assumed acceleration [m/s^2] (default: 5.0)")
    p_sim.add_argument("--n-points", type=int, default=16,
                       help="Number of baseline control points")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "df-trim":
        cmd_df_trim(args)
    elif args.command == "df-optimize":
        cmd_df_optimize(args)
    elif args.command == "df-baseline":
        cmd_df_baseline(args)
    elif args.command == "df-compare":
        cmd_df_compare(args)
    elif args.command == "df-simulate":
        cmd_df_simulate(args)


if __name__ == "__main__":
    main()
