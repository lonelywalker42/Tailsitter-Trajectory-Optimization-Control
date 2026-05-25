"""Custom training callbacks for progress display."""

import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Display episode reward, average reward, and training progress.

    Shows per-episode: reward, 20-episode rolling average, best reward,
    cumulative steps, progress bar, completion %, and ETA.
    """

    def __init__(self, total_timesteps: int, avg_window: int = 20, print_freq: int = 1):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.avg_window = avg_window
        self.print_freq = print_freq
        self.t_start = None
        self.best_reward = -np.inf
        self.episode_count = 0
        self._recent_rewards = deque(maxlen=avg_window)
        self._header_interval = 20

    def _on_training_start(self) -> None:
        self.t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._on_episode_end(info["episode"])
        return True

    def _on_episode_end(self, episode_info: dict) -> None:
        self.episode_count += 1
        ep_reward = episode_info["r"]

        self._recent_rewards.append(ep_reward)
        avg_reward = np.mean(self._recent_rewards)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward

        if self.episode_count % self.print_freq != 0:
            return

        # Progress
        total_steps = self.num_timesteps
        elapsed = time.time() - self.t_start
        pct = 100.0 * total_steps / self.total_timesteps if self.total_timesteps > 0 else 0
        steps_per_sec = total_steps / max(elapsed, 0.001)
        remaining = (self.total_timesteps - total_steps) / max(steps_per_sec, 0.001)

        # Progress bar
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "=" * filled + "-" * (bar_width - filled)

        # Print header
        if self.episode_count == 1 or self.episode_count % self._header_interval == 0:
            print(f"\n{'Episode':<10}{'Reward':>10}{'AvgRwd':>10}{'BestRwd':>10}"
                  f"{'Steps':>8}  {'Progress':<32}{'%':>6}{'ETA':>10}")
            print("-" * 96)

        print(f"{self.episode_count:<10}{ep_reward:>+10.1f}{avg_reward:>+10.1f}"
              f"{self.best_reward:>+10.1f}{total_steps:>8d}  "
              f"[{bar:<30}] {pct:>5.1f}%{_fmt_time(remaining):>10}")

        # Convergence indicators
        if avg_reward > 1e12:
            print("  ** Average reward > 1e12 -- training converging **")
        if ep_reward > 8000:
            print("  ** Episode reward > 8000 -- agent saved **")

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.t_start
        avg = np.mean(self._recent_rewards) if self._recent_rewards else 0
        print(f"\nTraining complete: {self.episode_count} episodes, "
              f"{self.num_timesteps} steps in {_fmt_time(elapsed)}")
        print(f"Final avg reward ({self.avg_window}-ep window): {avg:+.1f}")


def _fmt_time(seconds: float) -> str:
    """Format seconds to h:mm:ss or m:ss."""
    if seconds < 0:
        return "--:--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"
