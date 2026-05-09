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
    interval: int = 1200,
    repeat: bool = True,
    root_marker: bool = True,
):
    """
    Plot f(x) and animate the iteration points xs over it.

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


def plot_variable_traces(
    traces,
    *,
    title: str = "Iteration Traces",
    frames: int = 200,
    interval: int = 1200,
    repeat: bool = True,
    var_names=None,
):
    """
    Animate per-variable iteration evolution as split (stacked) subplots.
    Each subplot shows iteration index on the x-axis and the variable's value on the y-axis.

    Parameters
    ----------
    traces : list[list[float]]
        One list of iteration values per variable.
    title : str
        Figure-level title.
    frames, interval, repeat : animation params.
    var_names : optional list of axis labels (defaults to x, y, z, x4, ...).

    Returns
    -------
    fig, axes, ani
    """
    if not traces:
        raise ValueError("traces must be a non-empty list of per-variable iteration histories.")

    n = len(traces)
    iters_max = max(len(t) for t in traces)
    if iters_max == 0:
        raise ValueError("All traces are empty.")

    if var_names is None:
        defaults = ["x", "y", "z"]
        var_names = [defaults[i] if i < len(defaults) else f"x{i + 1}" for i in range(n)]

    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.4 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]

    lines = []
    points = []

    for i, ax in enumerate(axes):
        t = traces[i]
        idxs = list(range(len(t)))
        # static ghost path for context
        ax.plot(idxs, t, color="lightblue", linewidth=1, linestyle="--", label="full path")
        # converged value reference
        ax.axhline(t[-1], color="red", linewidth=0.8, linestyle=":", label=f"final = {t[-1]:.6f}")
        line, = ax.plot([], [], "b-o", linewidth=2, markersize=5)
        pt, = ax.plot([], [], "ro", markersize=9)
        lines.append(line)
        points.append(pt)
        ax.set_ylabel(var_names[i])
        ax.grid(True)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout()

    def update(frame):
        idx = (frame % iters_max) + 1
        artists = []
        for i in range(n):
            t = traces[i]
            j = min(idx, len(t))
            xs = list(range(j))
            ys = list(t[:j])
            lines[i].set_data(xs, ys)
            if j > 0:
                points[i].set_data([j - 1], [t[j - 1]])
            artists.extend([lines[i], points[i]])
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat)
    return fig, axes, ani


def plot_integration_method(
    f,
    a,
    b,
    n,
    integral_value,
    *,
    method: str = "trapezoidal",
    title: str = None,
    x_data=None,
    y_data=None,
    frames: int = None,
    interval: int = 900,
    repeat: bool = True,
):
    """
    Visualise a numerical-integration approximation with an animated
    sweep over the sample points.

    Parameters
    ----------
    f : callable or None
        Integrand (used when x_data/y_data are not supplied).
    a, b : float
        Integration limits.
    n : int
        Number of subintervals.
    integral_value : float
        The numerically computed integral.
    method : 'trapezoidal' or 'simpson'
    x_data, y_data : optional tabulated data (overrides f).
    frames, interval, repeat : animation params.

    Returns
    -------
    fig, ax, ani
    """
    if x_data is not None and y_data is not None:
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        a, b = float(x_data[0]), float(x_data[-1])
        n = len(x_data) - 1
        x_pts = x_data
        y_pts = y_data
        x_fine = np.linspace(a, b, 400)
        y_fine = np.interp(x_fine, x_data, y_data)
        f_label = "interpolated data"
    else:
        x_pts = np.linspace(a, b, n + 1)
        y_pts = _safe_eval_for_plot(f, x_pts)
        x_fine = np.linspace(a, b, 400)
        y_fine = _safe_eval_for_plot(f, x_fine)
        f_label = "f(x)"

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(x_fine, y_fine, color="blue", label=f_label, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True)

    # Pre-create all sub-region artists hidden, then reveal them frame by frame.
    patches = []  # for trapezoid: Polygon ; for simpson: (PolyCollection, Line2D)

    if method == "trapezoidal":
        for i in range(len(x_pts) - 1):
            poly = ax.fill(
                [x_pts[i], x_pts[i], x_pts[i + 1], x_pts[i + 1]],
                [0, y_pts[i], y_pts[i + 1], 0],
                color="orange", alpha=0.30, edgecolor="red", linewidth=1,
            )[0]
            poly.set_visible(False)
            patches.append(poly)
        method_pretty = "Composite Trapezoidal"
    elif method == "simpson":
        if n % 2 != 0:
            raise ValueError("Simpson plotting requires an even n.")
        for i in range(0, len(x_pts) - 1, 2):
            x0, x1, x2 = x_pts[i], x_pts[i + 1], x_pts[i + 2]
            y0, y1, y2 = y_pts[i], y_pts[i + 1], y_pts[i + 2]
            coeffs = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
            xs_seg = np.linspace(x0, x2, 60)
            ys_seg = np.polyval(coeffs, xs_seg)
            poly = ax.fill_between(xs_seg, 0, ys_seg, color="green", alpha=0.25)
            arc, = ax.plot(xs_seg, ys_seg, color="green", linewidth=1.2)
            poly.set_visible(False)
            arc.set_visible(False)
            patches.append((poly, arc))
        method_pretty = "Composite Simpson's"
    else:
        raise ValueError(f"Unknown method: {method}")

    sample_pts, = ax.plot([], [], "ro", markersize=6, label="Sample points")
    cur_pt, = ax.plot([], [], "o", color="yellow", markersize=12,
                      markeredgecolor="black", markeredgewidth=1.5, zorder=5)
    info = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_title(f"{title or method_pretty}    \u2248 {integral_value:.10f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()

    n_pts = len(x_pts)
    if frames is None:
        frames = max(60, n_pts * 4)

    def _patch_visible(k: int, idx: int) -> bool:
        # trapezoid k spans points k..k+1 (visible once we've reached point k+1)
        # simpson  k spans points 2k..2k+2 (visible once we've reached point 2k+2)
        if method == "trapezoidal":
            return idx >= k + 1
        return idx >= 2 * k + 2

    def update(frame):
        idx = frame % n_pts
        for k, p in enumerate(patches):
            vis = _patch_visible(k, idx)
            if isinstance(p, tuple):
                for q in p:
                    q.set_visible(vis)
            else:
                p.set_visible(vis)
        sample_pts.set_data(x_pts[: idx + 1], y_pts[: idx + 1])
        cur_pt.set_data([x_pts[idx]], [y_pts[idx]])
        info.set_text(
            f"Sample {idx}/{n_pts - 1}\nx = {x_pts[idx]:.4f}\ny = {y_pts[idx]:.4f}"
        )
        artists = [sample_pts, cur_pt, info]
        for p in patches:
            if isinstance(p, tuple):
                artists.extend(p)
            else:
                artists.append(p)
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=interval,
                        repeat=repeat, blit=False)
    return fig, ax, ani


def plot_gauss_quadrature(
    f,
    a,
    b,
    points: int,
    integral_value,
    *,
    title: str = None,
    frames: int = None,
    interval: int = 1400,
    repeat: bool = True,
):
    """
    Visualise Gauss-Legendre quadrature: f(x) on [a,b] with the Gauss nodes
    revealed one at a time by an animated marker.

    Returns
    -------
    fig, ax, ani
    """
    x_fine = np.linspace(a, b, 400)
    y_fine = _safe_eval_for_plot(f, x_fine)

    if points == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif points == 2:
        xi = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif points == 3:
        k = np.sqrt(3.0 / 5.0)
        xi = np.array([-k, 0.0, k])
        w = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError("Only 1, 2, or 3 points supported.")

    x_mapped = 0.5 * (b - a) * xi + 0.5 * (a + b)
    y_at_nodes = _safe_eval_for_plot(f, x_mapped)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(x_fine, y_fine, color="blue", linewidth=2, label="f(x)")
    ax.fill_between(x_fine, 0, y_fine, color="blue", alpha=0.12,
                    label=f"\u222b f dx \u2248 {integral_value:.8f}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True)

    vlines, dots, annots = [], [], []
    for xm, yn, wt in zip(x_mapped, y_at_nodes, w):
        vline = ax.vlines(xm, 0, yn, color="red", linestyle="--", linewidth=1)
        dot, = ax.plot([xm], [yn], "ro", markersize=10)
        annot = ax.annotate(f"w={wt:.4f}", (xm, yn), textcoords="offset points",
                            xytext=(6, 8), fontsize=9, color="darkred")
        vline.set_visible(False)
        dot.set_visible(False)
        annot.set_visible(False)
        vlines.append(vline)
        dots.append(dot)
        annots.append(annot)

    cur, = ax.plot([], [], "o", color="yellow", markersize=16,
                   markeredgecolor="black", markeredgewidth=2, zorder=5)
    info = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_title(title or f"{points}-point Gauss-Legendre on [{a}, {b}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()

    n_nodes = len(x_mapped)
    if frames is None:
        frames = max(60, n_nodes * 6)

    def update(frame):
        idx = frame % n_nodes
        for k in range(n_nodes):
            vis = k <= idx
            vlines[k].set_visible(vis)
            dots[k].set_visible(vis)
            annots[k].set_visible(vis)
        cur.set_data([x_mapped[idx]], [y_at_nodes[idx]])
        info.set_text(
            f"Node {idx + 1}/{n_nodes}\n"
            f"x = {x_mapped[idx]:.6f}\n"
            f"f(x) = {y_at_nodes[idx]:.6f}\n"
            f"w = {w[idx]:.6f}"
        )
        return [*vlines, *dots, *annots, cur, info]

    ani = FuncAnimation(fig, update, frames=frames, interval=interval,
                        repeat=repeat, blit=False)
    return fig, ax, ani


def plot_romberg_convergence(
    R,
    *,
    title: str = "Romberg Convergence",
    frames: int = None,
    interval: int = 1200,
    repeat: bool = True,
):
    """
    Plot the diagonal R[k][k] of the Romberg table to visualise convergence,
    with the points and connecting line revealed step-by-step.

    Returns
    -------
    fig, ax, ani
    """
    if not R:
        raise ValueError("Romberg table is empty.")

    diag = [R[k][k] for k in range(len(R))]
    ks = list(range(len(R)))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Reference: ghost full path (very faint) so axes auto-scale correctly.
    ax.plot(ks, diag, color="lightblue", linewidth=1, linestyle="--", alpha=0.7,
            label="full path")
    ax.axhline(diag[-1], color="red", linewidth=0.8, linestyle=":",
               label=f"Final \u2248 {diag[-1]:.12f}")

    line, = ax.plot([], [], "b-", linewidth=2, label="R[k][k] (best estimate)")
    pts, = ax.plot([], [], "bo", markersize=8)
    cur, = ax.plot([], [], "o", color="yellow", markersize=14,
                   markeredgecolor="black", markeredgewidth=2, zorder=5)
    info = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_xlabel("k  (Romberg row index)")
    ax.set_ylabel("Estimate")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    n = len(ks)
    if frames is None:
        frames = max(60, n * 8)

    def update(frame):
        idx = frame % n
        line.set_data(ks[: idx + 1], diag[: idx + 1])
        pts.set_data(ks[: idx + 1], diag[: idx + 1])
        cur.set_data([ks[idx]], [diag[idx]])
        info.set_text(
            f"k = {idx}\nR[{idx}][{idx}] = {diag[idx]:.12f}"
        )
        return [line, pts, cur, info]

    ani = FuncAnimation(fig, update, frames=frames, interval=interval,
                        repeat=repeat, blit=False)
    return fig, ax, ani


def plot_numerical_derivative(
    x_data,
    y_data,
    x_targets,
    derivs,
    *,
    f=None,
    title: str = "Numerical Differentiation",
    frames: int = None,
    interval: int = 900,
    repeat: bool = True,
):
    """
    Plot the data points (or function) and the tangent line(s) implied by
    the numerical derivative at each x_target.

    Animation: data points are revealed one by one, then each tangent line
    appears at its target.

    Returns
    -------
    fig, ax, ani
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_targets = np.atleast_1d(np.asarray(x_targets, dtype=float))
    derivs = np.atleast_1d(np.asarray(derivs, dtype=float))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    if f is not None:
        pad = 0.1 * (x_data.max() - x_data.min() + 1e-9)
        x_fine = np.linspace(x_data.min() - pad, x_data.max() + pad, 400)
        y_fine = _safe_eval_for_plot(f, x_fine)
        ax.plot(x_fine, y_fine, color="blue", alpha=0.7, linewidth=2, label="f(x)")

    # Ghost full data so axes auto-scale immediately.
    ax.plot(x_data, y_data, "o", color="lightsteelblue", markersize=5, alpha=0.6)

    data_pts, = ax.plot([], [], "bo", markersize=7, label="Data points")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True)

    span = (x_data.max() - x_data.min()) or 1.0
    colors = ["red", "green", "purple", "orange", "magenta", "brown"]
    tangent_lines = []
    tangent_marks = []
    for k, (xt, dt) in enumerate(zip(x_targets, derivs)):
        i = int(np.argmin(np.abs(x_data - xt)))
        yt = float(y_data[i])
        xs = np.linspace(xt - 0.30 * span, xt + 0.30 * span, 50)
        ys = yt + dt * (xs - xt)
        c = colors[k % len(colors)]
        line, = ax.plot(xs, ys, color=c, linewidth=2,
                        label=f"f'({xt:.4f}) \u2248 {dt:.6f}")
        mark, = ax.plot([xt], [yt], marker="*", color=c, markersize=14)
        line.set_visible(False)
        mark.set_visible(False)
        tangent_lines.append(line)
        tangent_marks.append(mark)

    cur, = ax.plot([], [], "o", color="yellow", markersize=14,
                   markeredgecolor="black", markeredgewidth=2, zorder=5)
    info = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    n_data = len(x_data)
    n_tan = len(tangent_lines)
    total = n_data + n_tan
    if frames is None:
        frames = max(60, total * 4)

    def update(frame):
        idx = frame % total
        if idx < n_data:
            i = idx
            data_pts.set_data(x_data[: i + 1], y_data[: i + 1])
            for k in range(n_tan):
                tangent_lines[k].set_visible(False)
                tangent_marks[k].set_visible(False)
            cur.set_data([x_data[i]], [y_data[i]])
            info.set_text(
                f"Data point {i + 1}/{n_data}\n"
                f"x = {x_data[i]:.4f}\n"
                f"y = {y_data[i]:.4f}"
            )
        else:
            data_pts.set_data(x_data, y_data)
            tj = idx - n_data
            for k in range(n_tan):
                vis = k <= tj
                tangent_lines[k].set_visible(vis)
                tangent_marks[k].set_visible(vis)
            xt = float(x_targets[tj])
            i = int(np.argmin(np.abs(x_data - xt)))
            yt = float(y_data[i])
            cur.set_data([xt], [yt])
            info.set_text(
                f"Tangent {tj + 1}/{n_tan}\n"
                f"x = {xt:.4f}\n"
                f"f'(x) \u2248 {float(derivs[tj]):.6f}"
            )
        return [data_pts, cur, info, *tangent_lines, *tangent_marks]

    ani = FuncAnimation(fig, update, frames=frames, interval=interval,
                        repeat=repeat, blit=False)
    return fig, ax, ani
