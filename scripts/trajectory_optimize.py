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
    parser.add_argument(
        "--two-stage", action="store_true",
        help="Use two-stage solve: coarse (N=20) then fine (N=num-nodes)")
    parser.add_argument(
        "--mu-strategy", type=str, default="adaptive",
        choices=["adaptive", "monotone"],
        help="IPOPT barrier update strategy (default: adaptive)")
    parser.add_argument(
        "--hessian", type=str, default="exact",
        choices=["exact", "limited-memory"],
        help="IPOPT Hessian approximation (default: exact)")
    parser.add_argument(
        "--collocation-degree", type=int, default=1, choices=[1, 2, 3],
        help="Radau collocation degree (default: 1)")
    args = parser.parse_args()

    import numpy as np

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    config = TrajectoryOptConfig(
        direction=args.direction,
        num_nodes=args.num_nodes,
        tf_guess=args.tf_guess,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_print_level=args.ipopt_print_level,
        ipopt_mu_strategy=args.mu_strategy,
        ipopt_hessian_approximation=args.hessian,
        collocation_degree=args.collocation_degree,
    )

    total_time = 0.0

    if args.two_stage:
        # Stage 1: Coarse solve
        print("=" * 50)
        print("Stage 1: Coarse solve (N=20)")
        print("=" * 50)
        coarse_config = TrajectoryOptConfig(
            direction=args.direction,
            num_nodes=20,
            tf_guess=args.tf_guess,
            ipopt_max_iter=500,
            ipopt_print_level=args.ipopt_print_level,
            ipopt_mu_strategy=args.mu_strategy,
            ipopt_hessian_approximation=args.hessian,
            collocation_degree=args.collocation_degree,
        )
        coarse_opt = TailsitterTrajectoryOptimizer(coarse_config)
        coarse_opt.build()
        coarse_result = coarse_opt.solve()
        total_time += coarse_result.solve_time

        print(f"  Solve time:  {coarse_result.solve_time:.1f} s")
        print(f"  Final time:  {coarse_result.tf:.3f} s")
        print(f"  Cost:        {coarse_result.cost:.4f}")
        print(f"  Success:     {coarse_result.success}")

        # Stage 2: Fine solve (warm-started)
        print()
        print("=" * 50)
        print(f"Stage 2: Fine solve (N={args.num_nodes}, warm-started)")
        print("=" * 50)

        fine_config = TrajectoryOptConfig(
            direction=args.direction,
            num_nodes=args.num_nodes,
            tf_guess=coarse_result.tf,
            ipopt_max_iter=args.ipopt_max_iter,
            ipopt_print_level=args.ipopt_print_level,
            ipopt_mu_strategy=args.mu_strategy,
            ipopt_hessian_approximation=args.hessian,
            ipopt_warm_start=True,
            collocation_degree=args.collocation_degree,
        )
        fine_opt = TailsitterTrajectoryOptimizer(fine_config)
        fine_opt.build()

        w0_fine = TailsitterTrajectoryOptimizer.resample_solution(
            coarse_result, N_new=args.num_nodes)
        result = fine_opt.solve_from_guess(w0_fine)
        total_time += result.solve_time
    else:
        # Single-stage solve
        print(f"Direction: {args.direction}")
        print(f"Mesh nodes: {args.num_nodes}")
        print(f"Building optimization problem...")

        optimizer = TailsitterTrajectoryOptimizer(config)
        optimizer.build()

        print(f"Solving with IPOPT...")
        result = optimizer.solve()
        total_time = result.solve_time

    print(f"\n{'='*50}")
    print(f"Total time:  {total_time:.1f} s")
    print(f"Solve time:  {result.solve_time:.1f} s")
    print(f"Final time:  {result.tf:.3f} s")
    print(f"Cost:        {result.cost:.4f}")
    print(f"Success:     {result.success}")
    print(f"{'='*50}")

    # Save results
    suffix = f"_N{args.num_nodes}"
    if args.two_stage:
        suffix += "_twostage"
    npz_path = output_dir / f"trajopt_{args.direction}{suffix}.npz"
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
