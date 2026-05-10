import numpy as np


def double_trapz(f, a, b, c, d, nx, ny, *, print_table: bool = True):
    """Composite 2D Trapezoidal Rule for f(x, y) on the rectangle [a, b] x [c, d].

    Builds a uniform (nx x ny) grid, evaluates f on it, and applies the
    weighted-sum form of the 2D trapezoidal rule:

        I  =  (hx * hy / 4) * ( C  +  2*B  +  4*I_in )

    where
        C    = sum of the 4 corner values of Z
        B    = sum of the 4 edge interiors  (top, bottom, left, right)
        I_in = sum of all strictly-interior values

    Parameters
    ----------
    f : callable
        Vectorised function f(X, Y) accepting numpy arrays.
    a, b : float
        Bounds in x  (b > a).
    c, d : float
        Bounds in y  (d > c).
    nx, ny : int
        Number of grid points in each direction (>= 2).
    print_table : bool, default True
        If True, print the domain, grid, Z matrix, and a step-by-step
        formula breakdown to stdout.

    Returns
    -------
    float
        Approximation of  integral over [a,b] x [c,d] of  f(x, y) dx dy.
    """
    # ---- 1. Validate inputs -------------------------------------------------
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be >= 2.")
    if b <= a or d <= c:
        raise ValueError("Need b > a and d > c.")

    # ---- 2. Build the uniform grid and evaluate Z = f(X, Y) ----------------
    x = np.linspace(a, b, nx)
    y = np.linspace(c, d, ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="ij")  # rows = x, cols = y
    Z = np.asarray(f(X, Y), dtype=float)

    # ---- 3. Weighted sums:  corners, edge interiors, strict interior -------
    corners = float(Z[0, 0] + Z[0, -1] + Z[-1, 0] + Z[-1, -1])
    boundaries = float(
        np.sum(Z[0, 1:-1]) + np.sum(Z[-1, 1:-1])      # top + bottom edges
        + np.sum(Z[1:-1, 0]) + np.sum(Z[1:-1, -1])    # left + right edges
    )
    interior = float(np.sum(Z[1:-1, 1:-1]))

    # ---- 4. Apply the formula ----------------------------------------------
    integral = (hx * hy / 4.0) * (corners + 2 * boundaries + 4 * interior)

    # ---- 5. Optional human-readable report ---------------------------------
    if print_table:
        print(f"Domain : x in [{a}, {b}], y in [{c}, {d}]")
        print(f"Grid   : nx = {nx}, ny = {ny}  (hx = {hx:.6g}, hy = {hy:.6g})")
        print()
        print(f"Z = f(x, y)  matrix  (shape {Z.shape[0]} x {Z.shape[1]}, rows = x, cols = y):")
        with np.printoptions(precision=4, suppress=True, linewidth=140,
                             threshold=200, edgeitems=4):
            print(Z)
        print()
        print("Formula (Composite 2D Trapezoidal Rule):")
        print("  I = (hx * hy / 4) * ( C + 2*B + 4*I_in )")
        print("    where")
        print("      C    = sum of the 4 corners")
        print("      B    = sum of the 4 edge interiors (top, bottom, left, right)")
        print("      I_in = sum of all strictly interior points")
        print()
        print("Plug in:")
        print(f"  C    = {corners:.10g}")
        print(f"  B    = {boundaries:.10g}")
        print(f"  I_in = {interior:.10g}")
        print(f"  hx   = {hx:.10g}")
        print(f"  hy   = {hy:.10g}")
        print(
            f"  I    = ({hx:.6g} * {hy:.6g} / 4) "
            f"* ({corners:.6g} + 2*{boundaries:.6g} + 4*{interior:.6g})"
        )
        print(f"       = {integral:.10f}")
        print()
        print(f"Volume \u2248 {integral:.10f}")

    return float(integral)


if __name__ == "__main__":
    # Example: integral over [0,2] x [0,3] of (x^2 + y^2) dx dy = 26 (analytic)
    double_trapz(lambda X, Y: X**2 + Y**2, a=0, b=2, c=0, d=3, nx=100, ny=100)
