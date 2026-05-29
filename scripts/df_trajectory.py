"""DF-based trajectory optimization script for tailsitter UAV transitions.

Replaces matlab/analysis/controller/trajectory_compare.m, Main.m, df_trim.m.

Usage:
    python scripts/df_trajectory.py df-trim
    python scripts/df_trajectory.py df-optimize --direction hover2forward
    python scripts/df_trajectory.py df-baseline
    python scripts/df_trajectory.py df-compare --direction hover2forward
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
)


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


if __name__ == "__main__":
    main()
