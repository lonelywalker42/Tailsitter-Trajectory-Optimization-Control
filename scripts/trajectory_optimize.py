"""Trajectory optimization script for tailsitter UAV transition flight.

Replaces matlab/trajectory/TraOptMain.m using CasADi + IPOPT.

Usage:
    python scripts/trajectory_optimize.py
    python scripts/trajectory_optimize.py --direction hover2forward
    python scripts/trajectory_optimize.py --num-nodes 200 --output-dir results/trajopt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tailsitter.trajectory_optimization import (
    TailsitterTrajectoryOptimizer,
    TrajectoryOptConfig,
)
from tailsitter.trajectory_plotting import plot_trajectory_result


def main():
    parser = argparse.ArgumentParser(
        description="Tailsitter UAV transition trajectory optimization")
    parser.add_argument(
        "--direction", type=str, default="forward2hover",
        choices=["forward2hover", "hover2forward"],
        help="Transition direction (default: forward2hover)")
    parser.add_argument(
        "--num-nodes", type=int, default=100,
        help="Number of mesh nodes (default: 100)")
    parser.add_argument(
        "--output-dir", type=str, default="results/trajopt",
        help="Output directory for results and plots (default: results/trajopt)")
    parser.add_argument(
        "--tf-guess", type=float, default=10.0,
        help="Initial guess for transition time in seconds (default: 10)")
    parser.add_argument(
        "--ipopt-max-iter", type=int, default=3000,
        help="IPOPT maximum iterations (default: 3000)")
    parser.add_argument(
        "--ipopt-print-level", type=int, default=5,
        help="IPOPT print level (default: 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    config = TrajectoryOptConfig(
        direction=args.direction,
        num_nodes=args.num_nodes,
        tf_guess=args.tf_guess,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_print_level=args.ipopt_print_level,
    )

    # Build and solve
    print(f"Direction: {args.direction}")
    print(f"Mesh nodes: {args.num_nodes}")
    print(f"Building optimization problem...")

    optimizer = TailsitterTrajectoryOptimizer(config)
    optimizer.build()

    print(f"Solving with IPOPT...")
    result = optimizer.solve()

    print(f"\n{'='*50}")
    print(f"Solve time:  {result.solve_time:.1f} s")
    print(f"Final time:  {result.tf:.3f} s")
    print(f"Cost:        {result.cost:.4f}")
    print(f"Success:     {result.success}")
    print(f"{'='*50}")

    # Save results
    import numpy as np
    npz_path = output_dir / f"trajopt_{args.direction}_N{args.num_nodes}.npz"
    np.savez(
        npz_path,
        time=result.time,
        state=result.state,
        control=result.control,
        tf=result.tf,
        cost=result.cost,
        solve_time=result.solve_time,
    )
    print(f"Results saved to {npz_path}")

    # Plot
    saved_plots = plot_trajectory_result(result, save_dir=output_dir)
    for p in saved_plots:
        print(f"Plot saved to {p}")


if __name__ == "__main__":
    main()
