"""Trim analysis script for tailsitter UAV.

Replaces matlab/trim/Main.m and matlab/trim/CFG2_main.m.
Computes transition corridor, level flight trim, and optional linearization.

Usage:
    python scripts/trim.py corridor
    python scripts/trim.py corridor --config aero_cfg1 --v-step 2 --theta-step 10
    python scripts/trim.py level-flight --v-step 0.5
    python scripts/trim.py point --V 15 --theta 10
    python scripts/trim.py corridor --linearize
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tailsitter.trim import (
    load_aero_data, sweep_corridor, sweep_level_flight,
    trim_at_conditions, trim_level_flight,
)
from tailsitter.trim_plotting import (
    plot_corridor, plot_eigenvalue_corridor, plot_level_flight,
)


def cmd_corridor(args):
    """Run transition corridor sweep."""
    aero = load_aero_data(args.config, args.data_dir)

    V_range = np.arange(args.v_min, args.v_max + args.v_step / 2, args.v_step)
    theta_range = np.arange(args.theta_min, args.theta_max + args.theta_step / 2,
                            args.theta_step)

    print(f"Config: {args.config}, xs={args.xs}")
    print(f"V range: {V_range[0]:.1f} to {V_range[-1]:.1f} m/s ({len(V_range)} points)")
    print(f"Theta range: {theta_range[0]:.1f} to {theta_range[-1]:.1f} deg "
          f"({len(theta_range)} points)")
    print(f"Total grid: {len(V_range) * len(theta_range)} points")
    print(f"Max restarts: {args.max_restarts}, Parallel jobs: {args.n_jobs}")
    print()

    t0 = time.time()
    mesh = sweep_corridor(V_range, theta_range, aero, xs=args.xs,
                          max_restarts=args.max_restarts,
                          n_jobs=args.n_jobs)
    elapsed = time.time() - t0
    print(f"\nCorridor sweep completed in {elapsed:.1f} s")

    # Linearization
    if args.linearize:
        from tailsitter.linearization import sweep_linearization
        print("\nRunning linearization analysis...")
        t0 = time.time()
        mesh = sweep_linearization(mesh, aero, xs=args.xs,
                                   n_jobs=args.n_jobs)
        elapsed = time.time() - t0
        print(f"Linearization completed in {elapsed:.1f} s")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"corridor_{args.config}_xs{args.xs}.npz"
    save_dict = {
        'trim_V': mesh.trim_V,
        'trim_theta': mesh.trim_theta,
        'trim_flag': mesh.trim_flag,
        'throttle': mesh.throttle,
        'elevator': mesh.elevator,
        'alpha': mesh.alpha,
        'gamma': mesh.gamma,
        'max_moment_ft': mesh.max_moment_ft,
        'max_moment_bt': mesh.max_moment_bt,
    }
    if mesh.real_max is not None:
        save_dict.update({
            'real_max': mesh.real_max,
            'real_min': mesh.real_min,
            'real_num': mesh.real_num,
            'complex_num': mesh.complex_num,
            'controllability_index': mesh.controllability_index,
            'min_singular_value': mesh.min_singular_value,
        })
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    # Plot
    saved_plots = plot_corridor(mesh, output_dir)
    for p in saved_plots:
        print(f"Plot saved to {p}")

    if args.linearize and mesh.real_max is not None:
        saved_plots = plot_eigenvalue_corridor(mesh, output_dir)
        for p in saved_plots:
            print(f"Plot saved to {p}")


def cmd_level_flight(args):
    """Run level flight trim sweep."""
    aero = load_aero_data(args.config, args.data_dir)

    V_range = np.arange(args.v_min, args.v_max + args.v_step / 2, args.v_step)

    print(f"Config: {args.config}, xs={args.xs}")
    print(f"V range: {V_range[0]:.1f} to {V_range[-1]:.1f} m/s ({len(V_range)} points)")
    print()

    t0 = time.time()
    result = sweep_level_flight(V_range, aero, xs=args.xs)
    elapsed = time.time() - t0
    print(f"\nLevel flight sweep completed in {elapsed:.1f} s")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"level_flight_{args.config}_xs{args.xs}.npz"
    np.savez(npz_path, **result)
    print(f"Results saved to {npz_path}")

    # Plot
    saved_plots = plot_level_flight(result, output_dir)
    for p in saved_plots:
        print(f"Plot saved to {p}")


def cmd_point(args):
    """Run single point trim."""
    aero = load_aero_data(args.config, args.data_dir)

    print(f"Config: {args.config}, xs={args.xs}")
    print(f"V={args.V} m/s, theta={args.theta} deg")
    print()

    tp = trim_at_conditions(args.V, args.theta, aero, xs=args.xs,
                            max_restarts=args.max_restarts)

    if tp.trim_flag:
        print(f"Trim found:")
        print(f"  Throttle:    {tp.throttle:.4f}")
        print(f"  Elevator:    {tp.elevator_deg:.2f} deg")
        print(f"  Alpha:       {tp.alpha_deg:.2f} deg")
        print(f"  Gamma:       {tp.gamma_deg:.2f} deg")
        print(f"  Residual:    {tp.residual:.6e}")

        # Also try linearization
        if args.linearize:
            from tailsitter.linearization import linearize_at_point, analyze_system
            A, B = linearize_at_point(args.V, tp.alpha_deg, args.theta,
                                      tp.throttle, tp.elevator_deg, aero, args.xs)
            analysis = analyze_system(A, B)
            print(f"\nLinearization:")
            print(f"  Eigenvalues: {analysis.eigen_info.eigenvalues}")
            print(f"  Stability: {analysis.stability_info.type}")
            print(f"  Controllable: {analysis.controllability_info.is_fully_controllable}")
    else:
        print("Trim not found at this operating point.")


def main():
    parser = argparse.ArgumentParser(
        description="Tailsitter UAV trim analysis")
    parser.add_argument(
        "--config", type=str, default="aero_cfg2",
        help="Aero configuration name (default: aero_cfg2)")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Path to processed data directory (default: data/processed)")
    parser.add_argument(
        "--xs", type=float, default=0.0,
        help="Moving mass position in meters (default: 0)")
    parser.add_argument(
        "--output-dir", type=str, default="results/trim",
        help="Output directory (default: results/trim)")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # corridor subcommand
    p_corridor = subparsers.add_parser("corridor",
                                       help="Run transition corridor sweep")
    p_corridor.add_argument("--v-min", type=float, default=0.0)
    p_corridor.add_argument("--v-max", type=float, default=20.0)
    p_corridor.add_argument("--v-step", type=float, default=1.0)
    p_corridor.add_argument("--theta-min", type=float, default=0.0)
    p_corridor.add_argument("--theta-max", type=float, default=90.0)
    p_corridor.add_argument("--theta-step", type=float, default=5.0)
    p_corridor.add_argument("--max-restarts", type=int, default=15)
    p_corridor.add_argument("--n-jobs", type=int, default=1,
                            help="Number of parallel workers (default: 1)")
    p_corridor.add_argument("--linearize", action="store_true",
                            help="Run linearization analysis at trim points")

    # level-flight subcommand
    p_lf = subparsers.add_parser("level-flight",
                                 help="Run level flight trim sweep")
    p_lf.add_argument("--v-min", type=float, default=0.0)
    p_lf.add_argument("--v-max", type=float, default=20.0)
    p_lf.add_argument("--v-step", type=float, default=0.5)

    # point subcommand
    p_point = subparsers.add_parser("point",
                                    help="Single point trim")
    p_point.add_argument("--V", type=float, required=True,
                         help="Airspeed (m/s)")
    p_point.add_argument("--theta", type=float, required=True,
                         help="Pitch angle (degrees)")
    p_point.add_argument("--max-restarts", type=int, default=15)
    p_point.add_argument("--linearize", action="store_true",
                         help="Also run linearization")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "corridor":
        cmd_corridor(args)
    elif args.command == "level-flight":
        cmd_level_flight(args)
    elif args.command == "point":
        cmd_point(args)


if __name__ == "__main__":
    main()
