"""Result visualization utilities.

Replaces RL/result/TrainPlot.m and RL/result/RLsim_plot.m.
SCI journal (Nature) style: serif fonts, muted color palette, clean layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
from contextlib import contextmanager

# ── Nature journal color palette ──────────────────────────────────────────────
COLORS = {
    "blue": "#3B7DD8",
    "orange": "#E8922F",
    "teal": "#2CA08E",
    "red": "#D64045",
    "gray": "#666666",
    "light_gray": "#CCCCCC",
}


@contextmanager
def _sci_style():
    """Temporarily apply SCI journal rcParams."""
    original = {
        k: mpl.rcParams[k]
        for k in [
            "font.family",
            "font.size",
            "axes.linewidth",
            "axes.grid",
            "grid.linewidth",
            "grid.alpha",
            "grid.color",
            "xtick.major.width",
            "ytick.major.width",
            "xtick.major.size",
            "ytick.major.size",
            "xtick.labelsize",
            "ytick.labelsize",
            "axes.labelsize",
            "axes.titlesize",
            "legend.fontsize",
            "figure.dpi",
            "savefig.dpi",
            "lines.linewidth",
        ]
    }
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "grid.color": "#CCCCCC",
            "grid.linestyle": "--",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "lines.linewidth": 1.5,
        }
    )
    try:
        yield
    finally:
        mpl.rcParams.update(original)


def _box_on(ax, linewidth: float = 1.0):
    """Apply MATLAB-style 'box on' to an axes (visible frame border)."""
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


# ── Static plots ──────────────────────────────────────────────────────────────


def plot_reward_curve(
    episode_rewards: np.ndarray,
    avg_rewards: np.ndarray,
    save_path: str,
):
    """Plot episode rewards and running average (SCI style).

    Replaces TrainPlot.m.
    """
    with _sci_style():
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(
            episode_rewards,
            color=COLORS["light_gray"],
            linewidth=0.5,
            label="Episode reward",
        )
        ax.plot(
            avg_rewards,
            color=COLORS["blue"],
            linewidth=1.2,
            label="Running average",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(frameon=False, loc="lower right")
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def plot_simulation_response(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Plot simulation response in 4x3 subplot grid with V-theta phase portrait.

    SCI (Nature) journal style.
    Replaces RLsim_plot.m.
    States: [u, w, V, alpha, theta, q, h]
    Controls: [dt, de]
    """
    V = states[:, 2]
    alpha = np.degrees(states[:, 3])
    theta = np.degrees(states[:, 4])
    h = states[:, 6]
    q = np.degrees(states[:, 5])
    dt = controls[:, 0]
    de = controls[:, 1]

    with _sci_style():
        fig = plt.figure(figsize=(10, 5.5))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)

        # ── Left: V-theta phase portrait (spans rows 0-2, col 0) ──
        ax_phase = fig.add_subplot(gs[0:3, 0])
        ax_phase.plot(V, theta, color=COLORS["blue"], linewidth=0.9)
        # Mark start and end
        ax_phase.plot(V[0], theta[0], "o", color=COLORS["teal"], markersize=5, zorder=5)
        ax_phase.plot(V[-1], theta[-1], "s", color=COLORS["red"], markersize=5, zorder=5)
        ax_phase.set_xlabel(r"$V$ [m/s]")
        ax_phase.set_ylabel(r"$\theta$ [deg]")
        ax_phase.set_axisbelow(True)

        # ── Row 0, col 1: V(t) ──
        ax_v = fig.add_subplot(gs[0, 1])
        ax_v.plot(time, V, color=COLORS["blue"])
        ax_v.set_ylabel(r"$V$ [m/s]")
        ax_v.set_xticklabels([])

        # ── Row 1, col 1: alpha(t) ──
        ax_a = fig.add_subplot(gs[1, 1], sharex=ax_v)
        ax_a.plot(time, alpha, color=COLORS["teal"])
        ax_a.set_ylabel(r"$\alpha$ [deg]")
        ax_a.set_xticklabels([])

        # ── Row 2, col 1: theta(t) ──
        ax_t = fig.add_subplot(gs[2, 1], sharex=ax_v)
        ax_t.plot(time, theta, color=COLORS["orange"])
        ax_t.set_ylabel(r"$\theta$ [deg]")

        # ── Row 3, col 0-1: h(t) ──
        ax_h = fig.add_subplot(gs[3, 0:2])
        ax_h.plot(time, h, color=COLORS["blue"])
        ax_h.set_xlabel(r"$t$ [s]")
        ax_h.set_ylabel(r"$h$ [m]")
        ax_h.set_axisbelow(True)

        # ── Col 2, rows 0-1: dt(t) ──
        ax_dt = fig.add_subplot(gs[0:2, 2])
        ax_dt.plot(time, dt, color=COLORS["red"])
        ax_dt.set_ylabel(r"$\delta_t$")
        ax_dt.set_xlabel(r"$t$ [s]")
        ax_dt.set_axisbelow(True)

        # ── Col 2, rows 2-3: de(t) ──
        ax_de = fig.add_subplot(gs[2:4, 2])
        ax_de.plot(time, de, color=COLORS["red"])
        ax_de.set_ylabel(r"$\delta_e$")
        ax_de.set_xlabel(r"$t$ [s]")
        ax_de.set_axisbelow(True)

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


# ── Flight animation ──────────────────────────────────────────────────────────


def _draw_tailsitter(ax, x, h, theta, scale=1.0, alpha=1.0):
    """Draw a simple tailsitter schematic at position (x, h) with pitch theta.

    Coordinate system: x horizontal, h vertical (up). Theta is pitch angle
    measured from horizontal (0 = level flight, pi/2 = hover nose-up).
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # Body dimensions (in local body frame, before rotation)
    body_len = 1.5 * scale
    wing_span = 1.8 * scale
    tail_span = 0.6 * scale

    # Body line (nose to tail)
    nose_local = np.array([body_len / 2, 0])
    tail_local = np.array([-body_len / 2, 0])

    # Wing (at ~1/3 from nose)
    wing_center = np.array([body_len / 6, 0])
    wing_left = wing_center + np.array([0, wing_span / 2])
    wing_right = wing_center + np.array([0, -wing_span / 2])

    # Tail (at rear)
    tail_top = tail_local + np.array([0, tail_span / 2])
    tail_bottom = tail_local + np.array([0, -tail_span / 2])

    # Rotation matrix
    R = np.array([[c, -s], [s, c]])

    def transform(p):
        return R @ p + np.array([x, h])

    # Draw body
    p_nose = transform(nose_local)
    p_tail = transform(tail_local)
    ax.plot(
        [p_nose[0], p_tail[0]],
        [p_nose[1], p_tail[1]],
        color=COLORS["gray"],
        linewidth=2.0,
        alpha=alpha,
        solid_capstyle="round",
    )

    # Draw wing
    p_wl = transform(wing_left)
    p_wr = transform(wing_right)
    ax.plot(
        [p_wl[0], p_wr[0]],
        [p_wl[1], p_wr[1]],
        color=COLORS["blue"],
        linewidth=2.5,
        alpha=alpha,
        solid_capstyle="round",
    )

    # Draw tail
    p_tt = transform(tail_top)
    p_tb = transform(tail_bottom)
    ax.plot(
        [p_tt[0], p_tb[0]],
        [p_tt[1], p_tb[1]],
        color=COLORS["blue"],
        linewidth=1.5,
        alpha=alpha,
        solid_capstyle="round",
    )

    # Draw propeller arc at nose
    prop_r = 0.3 * scale
    angles = np.linspace(0, 2 * np.pi, 20)
    prop_x = p_nose[0] + prop_r * np.cos(angles)
    prop_y = p_nose[1] + prop_r * np.sin(angles)
    ax.plot(prop_x, prop_y, color=COLORS["red"], linewidth=0.8, alpha=alpha * 0.7)


def animate_flight(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    save_path: str,
    title: str = "",
    fps: int = 20,
    dpi: int = 150,
) -> None:
    """Create a flight animation GIF showing aircraft attitude and position.

    Left panel: tailsitter schematic moving in x-h plane with trail.
    Right top: V(t) and theta(t) with moving time marker.
    Right bottom: throttle and elevator with moving time marker.

    States: [u, w, V, alpha, theta, q, h]
    Controls: [dt, de]
    """
    N = len(time)
    # Subsample for reasonable file size
    if N > 400:
        step = max(1, N // 250)
        idx = np.arange(0, N, step)
        # Always include last frame
        if idx[-1] != N - 1:
            idx = np.append(idx, N - 1)
    else:
        idx = np.arange(N)

    time_s = time[idx]
    states_s = states[idx]
    controls_s = controls[idx]
    N_s = len(idx)

    # Extract quantities
    u = states_s[:, 0]
    V = states_s[:, 2]
    theta = states_s[:, 4]
    h = states_s[:, 6]
    dt_ctrl = controls_s[:, 0]
    de_ctrl = controls_s[:, 1]

    # Compute horizontal position by integrating u
    dx = np.zeros(N_s)
    if N_s > 1:
        dt_arr = np.diff(time_s)
        dx[1:] = np.cumsum(u[:-1] * dt_arr)
    x_pos = dx  # relative horizontal displacement

    with _sci_style():
        fig = plt.figure(figsize=(10, 4.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.35)

        # ── Left: flight path ──
        ax_flight = fig.add_subplot(gs[:, 0])
        ax_flight.set_xlabel("$x$ [m]")
        ax_flight.set_ylabel("$h$ [m]")
        ax_flight.set_aspect("equal", adjustable="datalim")
        ax_flight.set_axisbelow(True)

        # Set axis limits with margin
        x_range = [np.min(x_pos) - 3, np.max(x_pos) + 3]
        h_range = [max(np.min(h) - 3, -5), np.max(h) + 5]
        ax_flight.set_xlim(x_range)
        ax_flight.set_ylim(h_range)

        # Ground line
        ax_flight.axhline(y=0, color=COLORS["light_gray"], linewidth=0.8, zorder=0)

        # Trail line (fading)
        trail, = ax_flight.plot([], [], "-", color=COLORS["light_gray"], linewidth=0.6, alpha=0.5)
        # Aircraft drawing (will be updated each frame)
        aircraft_lines = []

        # ── Right top: V and theta ──
        ax_state = fig.add_subplot(gs[0, 1])
        ax_state.plot(time_s, V, color=COLORS["blue"], linewidth=0.8, label=r"$V$")
        ax_state.plot(time_s, np.degrees(theta), color=COLORS["orange"], linewidth=0.8, label=r"$\theta$")
        ax_state.set_ylabel(r"$V$ [m/s], $\theta$ [deg]")
        ax_state.legend(frameon=False, loc="upper right", ncol=2)
        ax_state.set_xlabel(r"$t$ [s]")
        ax_state.set_axisbelow(True)
        vline_state = ax_state.axvline(x=time_s[0], color=COLORS["red"], linewidth=0.6, linestyle="-", alpha=0.7)

        # ── Right bottom: controls ──
        ax_ctrl = fig.add_subplot(gs[1, 1])
        ax_ctrl.plot(time_s, dt_ctrl, color=COLORS["red"], linewidth=0.8, label=r"$\delta_t$")
        ax_ctrl.plot(time_s, de_ctrl, color=COLORS["teal"], linewidth=0.8, label=r"$\delta_e$")
        ax_ctrl.set_xlabel(r"$t$ [s]")
        ax_ctrl.set_ylabel(r"$\delta_t$, $\delta_e$")
        ax_ctrl.legend(frameon=False, loc="upper right", ncol=2)
        ax_ctrl.set_axisbelow(True)
        vline_ctrl = ax_ctrl.axvline(x=time_s[0], color=COLORS["red"], linewidth=0.6, linestyle="-", alpha=0.7)

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")

        def _init():
            return [trail, vline_state, vline_ctrl]

        def _update(frame):
            # Update trail
            trail.set_data(x_pos[: frame + 1], h[: frame + 1])

            # Redraw aircraft at current position
            for line in aircraft_lines:
                line.remove()
            aircraft_lines.clear()

            # Draw aircraft
            _draw_tailsitter(
                ax_flight,
                x_pos[frame],
                h[frame],
                theta[frame],
                scale=max(0.5, min(2.0, (x_range[1] - x_range[0]) / 30)),
            )
            # Collect newly added lines (last 4: body, wing, tail, prop arc)
            aircraft_lines.extend(ax_flight.lines[-4:])

            # Update time markers
            vline_state.set_xdata([time_s[frame], time_s[frame]])
            vline_ctrl.set_xdata([time_s[frame], time_s[frame]])

            return [trail, vline_state, vline_ctrl] + aircraft_lines

        anim = FuncAnimation(
            fig,
            _update,
            frames=N_s,
            init_func=_init,
            blit=False,
            interval=1000 // fps,
        )

        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)


# ── Episode runner ────────────────────────────────────────────────────────────


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
    controls_list = [np.zeros(2)]  # No action at t=0
    time_list = [0.0]

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        states_list.append(env._state.copy())
        controls_list.append(action.copy())
        time_list.append(env._t)

        if terminated or truncated:
            break

    time = np.array(time_list)
    states = np.array(states_list)
    controls = np.array(controls_list)

    return time, states, controls
