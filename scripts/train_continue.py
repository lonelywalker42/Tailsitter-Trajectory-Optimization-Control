"""Continue training from a saved checkpoint.

Replaces RL/train_continue.m.

Usage:
    python scripts/train_continue.py --checkpoint results/DP_sac_40000000/agent --algo sac
    python scripts/train_continue.py --checkpoint results/DP_ppo_40000000/agent --algo ppo --timesteps 20000000
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from stable_baselines3 import SAC, PPO

from scripts.train import create_env


def main():
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved agent (without .zip extension)")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"],
                        help="Algorithm type")
    parser.add_argument("--timesteps", type=int, default=20000000,
                        help="Additional training timesteps")
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
        model = SAC.load(args.checkpoint, env=env)
    else:
        model = PPO.load(args.checkpoint, env=env)

    print(f"\nContinuing {args.algo.upper()} training for {args.timesteps} timesteps")
    print(f"Loaded from: {args.checkpoint}\n")

    # Train
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save
    run_name = f"DP_{args.algo}_continued_{args.timesteps}"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "agent"))
    print(f"\nAgent saved to {run_dir / 'agent.zip'}")


if __name__ == "__main__":
    main()
