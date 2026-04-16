import numpy as np

def gauss_seidel_method(A, b, x0=None, tol=1e-6, max_iter=100, print_table=True):
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)
    
    diag_dominant = all(2 * abs(A[i, i]) >= sum(abs(A[i, :])) for i in range(n))
    if print_table and not diag_dominant:
        print("Warning: Matrix is not strictly diagonally dominant; convergence not guaranteed.")
    
    history = [[0, x.copy(), np.nan]]
    if print_table:
        print(f"\nGauss-Seidel Method Iterations")
        print(f"{'Iter':<6} {'x':<30} {'Error':<15}")
        print("-" * 60)
        print(f"{0:<6} {np.array2string(x, precision=6, suppress_small=True):<30} {'-':<15}")
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (b[i] - sigma) / A[i, i]
        
        error = np.linalg.norm(x - x_old, ord=np.inf)
        history.append([k, x.copy(), error])
        
        if print_table:
            print(f"{k:<6} {np.array2string(x, precision=6, suppress_small=True):<30} {error:<15.8f}")
        
        if error < tol:
            if print_table:
                print(f"\nConverged in {k} iterations.")
                print(f"Solution: {np.array2string(x, precision=8, suppress_small=True)}")
                print(f"Residual norm: {np.linalg.norm(A @ x - b):.2e}")
            return x, k, True, history
    
    if print_table:
        print(f"\nDid not converge within {max_iter} iterations")
    return x, max_iter, False, history


if __name__ == "__main__":
    print("Gauss-Seidel Method - Linear System Solver")
    print("=" * 50)
    
    while True:
        try:
            n = int(input("Enter matrix size (2 for 2x2, 3 for 3x3): "))
            if n in [2, 3]:
                break
            else:
                print("Please enter 2 or 3.")
        except ValueError:
            print("Invalid input. Enter 2 or 3.")
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    print(f"\nEnter coefficients for each equation (format: a1 a2 ... an b):")
    for i in range(n):
        while True:
            try:
                coeffs = list(map(float, input(f"Equation {i+1}: ").split()))
                if len(coeffs) != n + 1:
                    print(f"Expected {n+1} numbers (coefficients + RHS). Try again.")
                    continue
                A[i, :] = coeffs[:n]
                b[i] = coeffs[-1]
                break
            except ValueError:
                print("Invalid numbers. Please enter space-separated numbers.")
    
    print("\nCoefficient matrix A:")
    print(A)
    print("RHS vector b:")
    print(b)
    
    tol = 1e-8
    max_iter = 100
    print_table = True
    
    print("\n" + "="*60)
    sol_gs, it_gs, conv_gs, _ = gauss_seidel_method(A, b, tol=tol, max_iter=max_iter, print_table=print_table)
    
    exact = np.linalg.solve(A, b)
    print("\n" + "="*60)
    print("Summary:")
    print(f"Gauss-Seidel solution: {np.array2string(sol_gs, precision=8, suppress_small=True)}  (iterations: {it_gs}, converged: {conv_gs})")
    print(f"Exact (numpy):         {np.array2string(exact, precision=8, suppress_small=True)}")