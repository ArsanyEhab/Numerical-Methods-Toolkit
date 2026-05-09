import numpy as np


def composite_trapezoidal(f=None, a=None, b=None, n=None,
                         x_data=None, y_data=None, print_table=True):
    """
    Composite Trapezoidal Rule

        ∫_a^b f(x) dx ≈ (h/2)[y0 + yn + 2*(y1 + y2 + ... + y_{n-1})]

    Use either a function (f, a, b, n) OR a table (x_data, y_data).

    Parameters
    ----------
    f : callable, optional
        Function to integrate.
    a, b : float, optional
        Integration limits.
    n : int, optional
        Number of subintervals.
    x_data, y_data : array_like, optional
        Tabulated data (must be equally spaced).
    print_table : bool
        Print the intermediate calculations.

    Returns
    -------
    integral : float
    """
    if x_data is not None and y_data is not None:
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
        h = x[1] - x[0]
        a, b = x[0], x[-1]
        n = len(x) - 1
    else:
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)

    interior_sum = np.sum(y[1:-1])
    S = y[0] + y[-1] + 2 * interior_sum
    integral = 0.5 * h * S

    if print_table:
        print("\n" + "=" * 60)
        print("           COMPOSITE TRAPEZOIDAL RULE")
        print("=" * 60)
        print(f"a = {a}, b = {b}, n = {n}, h = {h:.8f}")
        print(f"\nData points:")
        for i in range(len(x)):
            print(f"  x{i} = {x[i]:.6f}    y{i} = {y[i]:.8f}")
        print(f"\nFormula:  ∫ ≈ (h/2) * [y0 + yn + 2*(y1+...+y{n-1})]")
        print(f"        = ({h}/2) * [{y[0]:.8f} + {y[-1]:.8f} + 2*({interior_sum:.8f})]")
        print(f"        = {0.5*h:.8f} * {S:.8f}")
        print(f"        ≈ {integral:.10f}")
    return integral


def composite_simpson(f=None, a=None, b=None, n=None,
                     x_data=None, y_data=None, print_table=True):
    """
    Composite Simpson's 1/3 Rule (n must be even)

        ∫_a^b f(x) dx ≈ (h/3)[y0 + yn + 4*∑(odd indices) + 2*∑(even indices)]

    Use either a function (f, a, b, n) OR a table (x_data, y_data).

    Parameters
    ----------
    f : callable, optional
    a, b : float, optional
    n : int, optional
        Number of subintervals (must be even).
    x_data, y_data : array_like, optional
        Tabulated equally spaced data (number of points must be odd).
    print_table : bool

    Returns
    -------
    integral : float
    """
    if x_data is not None and y_data is not None:
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
        n = len(x) - 1
        if n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of intervals "
                             "(odd number of points).")
        h = x[1] - x[0]
        a, b = x[0], x[-1]
    else:
        if n % 2 != 0:
            raise ValueError("n must be even for Simpson's rule.")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)

    odd_sum = np.sum(y[1:-1:2])   # indices 1,3,5,...,n-1
    even_sum = np.sum(y[2:-1:2])  # indices 2,4,...,n-2
    S = y[0] + y[-1] + 4 * odd_sum + 2 * even_sum
    integral = (h / 3.0) * S

    if print_table:
        print("\n" + "=" * 60)
        print("           COMPOSITE SIMPSON'S RULE")
        print("=" * 60)
        print(f"a = {a}, b = {b}, n = {n}, h = {h:.8f}")
        print(f"\nData points:")
        for i in range(len(x)):
            print(f"  x{i} = {x[i]:.6f}    y{i} = {y[i]:.8f}")
        print(f"\nFormula:  ∫ ≈ (h/3) * [y0 + yn + 4*Σ(odd) + 2*Σ(even)]")
        print(f"        = ({h}/3) * [{y[0]:.8f} + {y[-1]:.8f} + 4*({odd_sum:.8f}) + 2*({even_sum:.8f})]")
        print(f"        = {h/3:.8f} * {S:.8f}")
        print(f"        ≈ {integral:.10f}")
    return integral


# ======================================================================
# Main interactive program
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("   NUMERICAL INTEGRATION (Trapezoidal & Simpson)")
    print("=" * 50)

    print("\nSelect method:")
    print("  1 - Composite Trapezoidal Rule")
    print("  2 - Composite Simpson's Rule")
    method = int(input("Enter choice (1/2): "))

    # Choose input type
    input_type = input("Use a function (f) or a table (t)? [f/t]: ").lower().strip()

    if input_type == 't':
        n_pts = int(input("How many data points? "))
        x_data = np.zeros(n_pts)
        y_data = np.zeros(n_pts)
        print("Enter x and y values (space separated, one pair per line):")
        for i in range(n_pts):
            vals = list(map(float, input(f"Point {i+1}: ").split()))
            x_data[i], y_data[i] = vals[0], vals[1]

        if method == 1:
            integral = composite_trapezoidal(x_data=x_data, y_data=y_data)
        elif method == 2:
            integral = composite_simpson(x_data=x_data, y_data=y_data)
        else:
            print("Invalid method choice.")
            exit()
        print(f"\nFinal integral ≈ {integral:.10f}")

    else:  # function input
        func_str = input("Enter f(x) as a lambda, e.g. lambda x: np.exp(x**2): ")
        f = eval(func_str)
        a = float(input("Lower limit a: "))
        b = float(input("Upper limit b: "))
        n = int(input("Number of subintervals n: "))

        if method == 1:
            integral = composite_trapezoidal(f=f, a=a, b=b, n=n)
        elif method == 2:
            if n % 2 != 0:
                print("Error: n must be even for Simpson's rule.")
                exit()
            integral = composite_simpson(f=f, a=a, b=b, n=n)
        else:
            print("Invalid method.")
            exit()
        print(f"\nFinal integral ≈ {integral:.10f}")