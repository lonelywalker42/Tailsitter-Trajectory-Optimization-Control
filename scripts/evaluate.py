"""Test trained agent and generate result plots.

Replaces RL/result/PocessTrainResult.m.

Usage:
    python scripts/evaluate.py --checkpoint results/DP_sac_40000000/agent --algo sac
    python scripts/evaluate.py --checkpoint results/DP_ppo_40000000/agent --algo ppo
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from stable_baselines3 import SAC, PPO

from scripts.train import create_env
from tailsitter.plotting import plot_simulation_response


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved agent (without .zip)")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--config-dir", type=str, default="config")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    data_dir = config_dir.parent / "data" / "processed"
    results_dir = Path(args.results_dir)

    # Create environment
    env = create_env(config_dir, data_dir, reset_mode=args.algo)

    # Load agent
    if args.algo == "sac":
        model = SAC.load(args.checkpoint)
    else:
        model = PPO.load(args.checkpoint)

    # Create output directory
    eval_dir = results_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Forward transition test: hover -> forward flight
    print("Running forward transition test...")
    forward_init = np.array([0.0, 0.01, np.pi / 2, 0.0, 0.0])
    forward_target = np.array([20.0, 4.9184 * np.pi / 180, 4.9184 * np.pi / 180])

    from tailsitter.plotting import run_evaluation_episode
    time_f, states_f, controls_f = run_evaluation_episode(
        env, model, forward_init, forward_target, cmd=-1
    )
    plot_simulation_response(
        time_f, states_f, controls_f,
        save_path=str(eval_dir / "forward_response.png"),
        title="Forward Transition (Hover -> Forward Flight)",
    )
    print(f"  Saved forward_response.png ({len(time_f)} steps)")

    # Back transition test: forward flight -> hover
    print("Running back transition test...")
    back_init = np.array([19.926, 1.715, 4.9184 * np.pi / 180, 0.0, 0.0])
    back_target = np.array([0.01, np.pi / 2, np.pi / 2])

    time_b, states_b, controls_b = run_evaluation_episode(
        env, model, back_init, back_target, cmd=1
    )
    plot_simulation_response(
        time_b, states_b, controls_b,
        save_path=str(eval_dir / "back_response.png"),
        title="Back Transition (Forward Flight -> Hover)",
    )
    print(f"  Saved back_response.png ({len(time_b)} steps)")

    print(f"\nResults saved to {eval_dir}")


if __name__ == "__main__":
    main()
