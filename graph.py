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
    frames: int = 200,
    interval: int = 400,
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

