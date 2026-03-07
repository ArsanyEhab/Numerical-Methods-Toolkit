import numpy as np

def secant_method(f, x0, x1, tol=1e-6, max_iter=100, print_table=True):
    """
    Secant Method: x_{n+1} = x_n - (x_n - x_{n-1})*f(x_n)/(f(x_n) - f(x_{n-1}))
    
    Parameters:
    f: function - equation f(x)=0
    x0, x1: float - initial guesses
    tol: float - tolerance for convergence
    max_iter: int - maximum number of iterations
    print_table: bool - whether to print iteration table
    
    Returns:
    root: float - approximated root
    iterations: int - number of iterations performed
    converged: bool - whether method converged
    history: list - list of [iteration, x, f(x)] for each step
    """
    
    x_prev, x_curr = x0, x1
    history = [[0, x_prev, f(x_prev)], [1, x_curr, f(x_curr)]]
    
    if print_table:
        print(f"\n{'Iteration':<10} {'x_n':<15} {'f(x_n)':<15} {'Error':<15}")
        print("-"*55)
        print(f"{0:<10} {x_prev:<15.8f} {f(x_prev):<15.8f} {'-':<15}")
        print(f"{1:<10} {x_curr:<15.8f} {f(x_curr):<15.8f} {'-':<15}")
    
    for i in range(2, max_iter + 1):
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        
        # Avoid division by zero
        if abs(f_curr - f_prev) < 1e-15:
            if print_table:
                print("⚠ Division by near-zero detected. Method stopped.")
            return x_curr, i, False, history
        
        x_next = x_curr - (x_curr - x_prev) * f_curr / (f_curr - f_prev)
        error = abs(x_next - x_curr)
        
        history.append([i, x_next, f(x_next)])
        
        if print_table:
            print(f"{i:<10} {x_next:<15.8f} {f(x_next):<15.8f} {error:<15.8f}")
        
        # Check convergence
        if error < tol or abs(f(x_next)) < tol:
            if print_table:
                print(f"\n✓ Converged to root: {x_next:.10f}")
                print(f"  Iterations: {i}")
                print(f"  f(root) = {f(x_next):.2e}")
            return x_next, i, True, history
        
        x_prev, x_curr = x_curr, x_next
    
    if print_table:
        print(f"\n⚠ Did not converge within {max_iter} iterations")
    
    return x_curr, max_iter, False, history