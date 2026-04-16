import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _safe_eval_for_plot(f, x_vals):
    try:
        return f(x_vals)
    except Exception:
        return np.vectorize(f)(x_vals)


def plot_iteration_trace(
    f,
    xs,
    *,
    title: str = "Iteration Trace",
    frames: int = 40,
    interval: int = 80,
    repeat: bool = True,
    root_marker: bool = True,
):
    """
    Plot f(x) and animate the iteration points xs over it.

    Parameters
    ----------
    f : callable
        Function f(x).
    xs : list[float]
        Iteration x-values (x0, x1, ...).
    title : str
        Plot title.
    frames : int
        Number of animation frames.
    interval : int
        Delay between frames in milliseconds.
    repeat : bool
        Whether the animation repeats.
    root_marker : bool
        If True, mark the last x in xs as the "root" estimate.

    Returns
    -------
    fig, ax, ani
    """
    if xs is None or len(xs) == 0:
        raise ValueError("xs must be a non-empty list of iteration values.")

    xs = list(xs)
    x_min = min(xs)
    x_max = max(xs)
    span = (x_max - x_min) or 1.0

    x_vals = np.linspace(x_min - 2 * span, x_max + 2 * span, 500)
    y_vals = _safe_eval_for_plot(f, x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, color="blue", label="f(x)")
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    point, = ax.plot([], [], "ro", markersize=8)
    trace, = ax.plot([], [], "orange", linewidth=2)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    trace_x = []
    trace_y = []

    if root_marker:
        root = xs[-1]
        y_root = f(root)
        ax.plot(root, y_root, "r*", label=f"Root = {root:.6f}")
        ax.text(root, y_root + 0.5, f"Root = {root:.6f}", color="black")

    def update(frame):
        index = frame % len(xs)

        if index == 0:
            trace_x.clear()
            trace_y.clear()

        x = xs[index]
        y = f(x)

        trace_x.append(x)
        trace_y.append(y)

        point.set_data([x], [y])
        trace.set_data(trace_x, trace_y)
        text.set_text(f"Iteration: {index}\nx = {x:.6f}")
        return point, trace, text

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat)
    return fig, ax, ani


def plot_phase_trajectory_2d(
    xs,
    ys,
    *,
    title: str = "Phase Trajectory",
    frames: int = 200,
    interval: int = 80,
    repeat: bool = True,
):
    """
    Animate the 2-D iteration trajectory (x_i, y_i).

    Returns
    -------
    fig, ax, ani
    """
    if not xs or not ys or len(xs) != len(ys):
        raise ValueError("xs and ys must be non-empty lists of equal length.")

    xs, ys = list(xs), list(ys)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # static full-path ghost so the user can see where the trajectory goes
    ax.plot(xs, ys, color="lightblue", linewidth=1, linestyle="--", label="full path")

    # mark start and end
    ax.plot(xs[0], ys[0], "gs", markersize=9, label=f"Start ({xs[0]:.4f}, {ys[0]:.4f})")
    ax.plot(xs[-1], ys[-1], "r*", markersize=12, label=f"End ({xs[-1]:.4f}, {ys[-1]:.4f})")
    ax.legend(fontsize=8)

    trace_line, = ax.plot([], [], "b-o", linewidth=2, markersize=5)
    text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(frame):
        idx = (frame % len(xs)) + 1
        if frame % len(xs) == 0:
            pass  # keep cumulative — looks better
        trace_line.set_data(xs[:idx], ys[:idx])
        text.set_text(f"Iter {idx - 1}:  x={xs[idx-1]:.6f}  y={ys[idx-1]:.6f}")
        return trace_line, text

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat)
    return fig, ax, ani


def plot_phase_trajectory_3d(
    xs,
    ys,
    zs,
    *,
    title: str = "Phase Trajectory (3D)",
    frames: int = 200,
    interval: int = 600,
    repeat: bool = True,
):
    """
    Animate the 3-D iteration trajectory (x_i, y_i, z_i).

    Returns
    -------
    fig, ax, ani
    """
    if not xs or not ys or not zs:
        raise ValueError("xs, ys, and zs must all be non-empty.")
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("xs, ys, and zs must have the same length.")

    xs, ys, zs = list(xs), list(ys), list(zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore[attr-defined]

    # static ghost path
    ax.plot(xs, ys, zs, color="lightblue", linewidth=1, linestyle="--", label="full path")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=80, marker="s",
               label=f"Start ({xs[0]:.3f}, {ys[0]:.3f}, {zs[0]:.3f})")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", s=120, marker="*",
               label=f"End ({xs[-1]:.3f}, {ys[-1]:.3f}, {zs[-1]:.3f})")
    ax.legend(fontsize=7)

    trace_line, = ax.plot([], [], [], "b-o", linewidth=2, markersize=5)
    text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=8)  # type: ignore[attr-defined]

    def update(frame):
        idx = (frame % len(xs)) + 1
        trace_line.set_data(xs[:idx], ys[:idx])
        trace_line.set_3d_properties(zs[:idx])
        text.set_text(
            f"Iter {idx-1}: x={xs[idx-1]:.4f}  y={ys[idx-1]:.4f}  z={zs[idx-1]:.4f}"
        )
        return trace_line, text

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat)
    return fig, ax, ani

