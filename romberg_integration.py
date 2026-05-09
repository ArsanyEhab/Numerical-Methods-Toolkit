import numpy as np

def romberg_integration(f, a, b, max_order=5, tol=1e-6, print_table=True):
    """
    Romberg integration using recursive Richardson extrapolation.
    Returns: integral, converged (bool), table (list of rows)
    """
    R = []
    h = b - a
    R.append([0.5 * h * (f(a) + f(b))])

    if print_table:
        print("\nRomberg Integration Table")
        print(f"{'k':<4} {'intervals':<10}", end="")
        for j in range(1, max_order+1):
            print(f"{'O(h^{})'.format(2*j):<18}", end="")
        print()
        print("-" * (20 + 18*max_order))

    for k in range(1, max_order):
        n_sub = 2**k
        h_new = (b - a) / n_sub
        # odd-indexed points: 1,3,5,... up to n_sub-1
        odd_points = np.array([a + (2*i - 1)*h_new for i in range(1, n_sub//2 + 1)])
        odd_sum = np.sum(f(odd_points))
        trap_new = 0.5 * R[k-1][0] + h_new * odd_sum
        new_row = [trap_new]

        for j in range(1, k+1):
            prev = new_row[j-1]
            above = R[k-1][j-1]
            extrap = prev + (prev - above) / (4**j - 1)
            new_row.append(extrap)

        R.append(new_row)

        if k >= 2:
            diff = abs(R[k][k] - R[k-1][k-1])
            if diff < tol:
                if print_table:
                    _print_romberg_table(R, max_order)
                    print(f"\nConverged after {k+1} rows.")
                return R[k][k], True, R

    if print_table:
        _print_romberg_table(R, max_order)
    return R[-1][-1], False, R


def _print_romberg_table(R, max_order):
    for k, row in enumerate(R):
        print(f"{k:<4} {2**k:<10}", end="")
        for val in row:
            print(f" {val:<17.10f}", end="")
        print()


if __name__ == "__main__":
    print("Romberg Integration")
    print("=" * 50)
    func_str = input("Enter integrand f(x) as lambda, e.g. lambda x: x**2 * np.exp(x): ")
    f = eval(func_str)
    a = float(input("Lower limit a: "))
    b = float(input("Upper limit b: "))
    max_order = int(input("Max extrapolation order (e.g. 5): ") or "5")
    tol = float(input("Tolerance (e.g. 1e-6): ") or "1e-6")

    I, conv, table = romberg_integration(f, a, b, max_order=max_order, tol=tol, print_table=True)
    print(f"\nRomberg integral = {I:.12f}  (converged: {conv})")