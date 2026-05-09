import numpy as np

def gauss_legendre_1pt(f, a, b):
    """1-point Gauss-Legendre (= midpoint rule)."""
    x_mid = 0.5 * (a + b)
    return (b - a) * f(x_mid)

def gauss_legendre_2pt(f, a, b):
    """2-point Gauss-Legendre quadrature on [a,b]."""
    xi = 1.0 / np.sqrt(3)
    x1 = 0.5 * (b - a) * (-xi) + 0.5 * (a + b)
    x2 = 0.5 * (b - a) * xi + 0.5 * (a + b)
    return 0.5 * (b - a) * (f(x1) + f(x2))

def gauss_legendre_3pt(f, a, b):
    """3-point Gauss-Legendre quadrature on [a,b]."""
    k = np.sqrt(3.0 / 5.0)
    w = np.array([5/9, 8/9, 5/9])
    xi = np.array([-k, 0.0, k])
    x_mapped = 0.5 * (b - a) * xi + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(x_mapped))


def gaussian_quadrature(f, a, b, points=2, print_table=True):
    """
    Gauss-Legendre quadrature with user-chosen number of points.

    Parameters
    ----------
    f : callable
    a, b : float
    points : int (1, 2, or 3)
    print_table : bool

    Returns
    -------
    integral : float
    """
    if points == 1:
        I = gauss_legendre_1pt(f, a, b)
        if print_table:
            print("\n1‑point Gauss (midpoint rule)")
            print(f"x_mid = {0.5*(a+b):.6f}, f(x_mid) = {f(0.5*(a+b)):.8f}")
    elif points == 2:
        I = gauss_legendre_2pt(f, a, b)
        if print_table:
            xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w = np.array([1.0, 1.0])
            _print_gauss_table(xi, w, a, b, f, "2‑point Gauss")
    elif points == 3:
        I = gauss_legendre_3pt(f, a, b)
        if print_table:
            k = np.sqrt(3/5)
            xi = np.array([-k, 0.0, k])
            w = np.array([5/9, 8/9, 5/9])
            _print_gauss_table(xi, w, a, b, f, "3‑point Gauss")
    else:
        raise NotImplementedError("Only 1, 2, or 3 points are supported.")
    return I


def _print_gauss_table(xi, w, a, b, f, title):
    print(f"\n{title}")
    print(f"{'Node ξ_i':<12} {'Weight w_i':<12} {'x_i = map(ξ_i)':<18} {'f(x_i)':<15}")
    print("-" * 60)
    for xi_i, w_i in zip(xi, w):
        x_i = 0.5*(b - a)*xi_i + 0.5*(a + b)
        print(f"{xi_i:<12.6f} {w_i:<12.6f} {x_i:<18.8f} {f(x_i):<15.8f}")
    print("-" * 60)


if __name__ == "__main__":
    print("Gaussian Quadrature")
    print("=" * 50)
    func_str = input("Enter integrand f(x) as lambda, e.g. lambda x: x**2 * np.exp(x): ")
    f = eval(func_str)
    a = float(input("Lower limit a: "))
    b = float(input("Upper limit b: "))
    print("\nChoose number of Gauss points:")
    print("  1 - 1‑point (midpoint)")
    print("  2 - 2‑point")
    print("  3 - 3‑point")
    pts = int(input("Your choice (1/2/3): "))

    I = gaussian_quadrature(f, a, b, points=pts, print_table=True)
    print(f"\nIntegral ≈ {I:.10f}")