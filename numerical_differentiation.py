import numpy as np

def numerical_derivative_from_data(x_data, y_data, x_target, h=None,
                                   method='auto', print_table=True):
    """
    Compute first derivative approximations at specified points.

    Parameters
    ----------
    x_data, y_data : array_like
        Tabulated data (must be equally spaced)
    x_target : float or array_like
        Point(s) at which to evaluate derivative
    h : float, optional
        Step size; inferred from x_data if None
    method : str
        'auto'       -> 3-pt endpoint / 2-pt central (best available)
        'forward2'   -> (f(x+h)-f(x))/h
        'backward2'  -> (f(x)-f(x-h))/h
        'central2'   -> (f(x+h)-f(x-h))/(2h)
        'forward3'   -> (-3f(x)+4f(x+h)-f(x+2h))/(2h)
        'backward3'  -> (3f(x)-4f(x-h)+f(x-2h))/(2h)
    print_table : bool

    Returns
    -------
    deriv : float or ndarray
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if h is None:
        h = x_data[1] - x_data[0]
        if not np.allclose(np.diff(x_data), h):
            raise ValueError("x_data must be equally spaced or provide h explicitly.")

    if np.isscalar(x_target):
        x_target = np.atleast_1d(x_target)
        scalar_out = True
    else:
        scalar_out = False

    deriv = np.zeros_like(x_target, dtype=float)

    if print_table:
        print(f"\nNumerical Differentiation using method: {method}")
        print(f"h = {h}")
        print(f"{'x':<10} {'f\'(x)':<15}")
        print("-"*25)

    for idx, x0 in enumerate(x_target):
        i = np.argmin(np.abs(x_data - x0))
        if not np.isclose(x_data[i], x0):
            raise ValueError(f"x_target={x0} not found in x_data.")

        if method == 'forward2':
            if i == len(x_data)-1:
                raise ValueError("forward2 requires a point ahead; x_target is at the end.")
            d = (y_data[i+1] - y_data[i]) / h
        elif method == 'backward2':
            if i == 0:
                raise ValueError("backward2 requires a point behind; x_target is at the start.")
            d = (y_data[i] - y_data[i-1]) / h
        elif method == 'central2':
            if i == 0 or i == len(x_data)-1:
                raise ValueError("central2 requires points on both sides; x_target is at boundary.")
            d = (y_data[i+1] - y_data[i-1]) / (2*h)
        elif method == 'forward3':
            if i > len(x_data)-3:
                raise ValueError("forward3 needs two points ahead.")
            f0, f1, f2 = y_data[i], y_data[i+1], y_data[i+2]
            d = (-3*f0 + 4*f1 - f2) / (2*h)
        elif method == 'backward3':
            if i < 2:
                raise ValueError("backward3 needs two points behind.")
            f0, fm1, fm2 = y_data[i], y_data[i-1], y_data[i-2]
            d = (3*f0 - 4*fm1 + fm2) / (2*h)
        elif method == 'auto':
            if i == 0:
                f0, f1, f2 = y_data[i], y_data[i+1], y_data[i+2]
                d = (-3*f0 + 4*f1 - f2) / (2*h)
            elif i == len(x_data)-1:
                f0, fm1, fm2 = y_data[i], y_data[i-1], y_data[i-2]
                d = (3*f0 - 4*fm1 + fm2) / (2*h)
            else:
                d = (y_data[i+1] - y_data[i-1]) / (2*h)
        else:
            raise ValueError(f"Unknown method: {method}")

        deriv[idx] = d
        if print_table:
            print(f"{x0:<10.4f} {d:<15.8f}")

    if print_table:
        print("-"*25)

    if scalar_out:
        return deriv[0]
    return deriv


if __name__ == "__main__":
    print("Numerical Differentiation")
    print("="*50)
    # Choose input type
    use_table = input("Use tabulated data? (y/n): ").lower().startswith('y')

    if use_table:
        n_pts = int(input("How many data points? "))
        x_data = np.zeros(n_pts)
        y_data = np.zeros(n_pts)
        print("Enter x and y values (space separated, one pair per line):")
        for i in range(n_pts):
            vals = list(map(float, input(f"Point {i+1}: ").split()))
            x_data[i], y_data[i] = vals[0], vals[1]
        # Ask which x_target(s) to evaluate
        targets = input("Enter x value(s) to compute derivative (space separated): ")
        x_target = np.array(list(map(float, targets.split())))
    else:
        func_str = input("Enter function f(x) as a lambda, e.g. lambda x: x**2: ")
        f = eval(func_str)
        x_target = float(input("Enter x at which to compute derivative: "))
        h = float(input("Enter step size h (e.g. 0.1): "))
        # We'll just use a tiny table around x_target
        x_data = np.array([x_target-h, x_target, x_target+h])
        y_data = np.array([f(x_target-h), f(x_target), f(x_target+h)])
        x_target = np.array([x_target])  # only one point
        use_table = True  # now we have a table

    print("\nAvailable methods:")
    print("  forward2   - 2‑point forward  O(h)")
    print("  backward2  - 2‑point backward O(h)")
    print("  central2   - 2‑point central  O(h²)")
    print("  forward3   - 3‑point forward  O(h²)")
    print("  backward3  - 3‑point backward O(h²)")
    print("  auto       - automatic best choice (3‑pt endpoint / 2‑pt central)")
    method = input("Choose method: ").strip().lower()

    deriv = numerical_derivative_from_data(x_data, y_data, x_target, method=method, print_table=True)
    print(f"\nResult: f' = {np.array2string(deriv, precision=8)}")