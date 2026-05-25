"""Result visualization utilities.

Replaces RL/result/TrainPlot.m and RL/result/RLsim_plot.m.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_reward_curve(
    episode_rewards: np.ndarray,
    avg_rewards: np.ndarray,
    save_path: str,
):
    """Plot episode rewards and running average.

    Replaces TrainPlot.m.
    """
    rgb = np.array([175, 224, 230]) / 255.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode_rewards, color=rgb, linewidth=0.5)
    ax.plot(avg_rewards, "k-", linewidth=1.2)
    ax.set_xlabel(r"$Episode\;Number$", fontsize=12)
    ax.set_ylabel(r"$Episode\;Reward$", fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_response(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Plot simulation response in 4x3 subplot grid with V-theta phase portrait.

    Replaces RLsim_plot.m.
    States: [u, w, V, alpha, theta, q, h]
    Controls: [dt, de]
    """
    r2d = 57.3  # rad to deg

    V = states[:, 2]
    alpha = np.degrees(states[:, 3])
    theta = np.degrees(states[:, 4])
    h = states[:, 6]
    dt = controls[:, 0]
    de = controls[:, 1]

    fig = plt.figure(figsize=(12, 6))

    # Large left panel: V-theta phase portrait
    ax_phase = fig.add_subplot(4, 3, (1, 10))
    ax_phase.plot(V, theta, "b-", linewidth=0.8)
    ax_phase.set_xlabel(r"$V\;[m/s]$", fontsize=12, fontweight="bold")
    ax_phase.set_ylabel(r"$\theta\;[^\circ]$", fontsize=12, fontweight="bold")
    ax_phase.set_xlim([0, 25])
    ax_phase.set_ylim([-10, 100])
    ax_phase.grid(True)

    # V time history
    ax_v = fig.add_subplot(4, 3, 2)
    ax_v.plot(time, V, "b-", linewidth=0.8)
    ax_v.set_ylabel(r"$V\;[m/s]$", fontsize=10, fontweight="bold")
    ax_v.set_ylim([0, 25])
    ax_v.grid(True)

    # alpha time history
    ax_a = fig.add_subplot(4, 3, 5)
    ax_a.plot(time, alpha, "b-", linewidth=0.8)
    ax_a.set_ylabel(r"$\alpha\;[^\circ]$", fontsize=10, fontweight="bold")
    ax_a.set_ylim([-10, 120])
    ax_a.grid(True)

    # theta time history
    ax_t = fig.add_subplot(4, 3, 8)
    ax_t.plot(time, theta, "b-", linewidth=0.8)
    ax_t.set_ylabel(r"$\theta\;[^\circ]$", fontsize=10, fontweight="bold")
    ax_t.set_ylim([-10, 120])
    ax_t.grid(True)

    # h time history
    ax_h = fig.add_subplot(4, 3, 11)
    ax_h.plot(time, h, "b-", linewidth=0.8)
    ax_h.set_ylabel(r"$h\;[m]$", fontsize=10, fontweight="bold")
    ax_h.set_xlabel(r"$t\;[s]$", fontsize=10, fontweight="bold")
    ax_h.grid(True)

    # dt time history
    ax_dt = fig.add_subplot(4, 3, (3, 6))
    ax_dt.plot(time, dt, "r-", linewidth=0.8)
    ax_dt.set_ylabel(r"$\delta_t$", fontsize=12, fontweight="bold")
    ax_dt.grid(True)

    # de time history
    ax_de = fig.add_subplot(4, 3, (9, 12))
    ax_de.plot(time, de, "r-", linewidth=0.8)
    ax_de.set_ylabel(r"$\delta_e$", fontsize=12, fontweight="bold")
    ax_de.set_xlabel(r"$t\;[s]$", fontsize=10, fontweight="bold")
    ax_de.grid(True)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation_episode(
    env,
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    cmd: int,
    max_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one episode with fixed initial conditions.

    Returns:
        time: [N] array
        states: [N, 7] array
        controls: [N, 2] array
    """
    # Reset with specific initial conditions
    obs, _ = env.reset()
    env._state = initial_state.copy()
    env._target = target.copy()
    env._cmd = cmd

    states_list = [initial_state.copy()]
    controls_list = []
    time_list = [0.0]

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        states_list.append(env._state.copy())
        controls_list.append(action.copy())
        time_list.append(env._t)

        if terminated or truncated:
            break

    time = np.array(time_list[:len(states_list)])
    states = np.array(states_list)
    controls = np.array(controls_list) if controls_list else np.zeros((0, 2))

    return time, states, controls
