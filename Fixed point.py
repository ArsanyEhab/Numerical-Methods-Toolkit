import numpy as np

def fixed_point_iteration(f, g, x0, tol=1e-6, max_iter=100, print_table=True):
    """
    Fixed Point Iteration Method: x_{n+1} = g(x_n)
    
    Parameters:
    f: function - original equation f(x)=0 (for verification)
    g: function - transformed equation x = g(x)
    x0: float - initial guess
    tol: float - tolerance for convergence
    max_iter: int - maximum number of iterations
    print_table: bool - whether to print iteration table
    
    Returns:
    root: float - approximated root
    iterations: int - number of iterations performed
    converged: bool - whether method converged
    history: list - list of [iteration, x, f(x)] for each step
    """
    
    x = x0
    history = [[0, x, f(x)]]
    
    if print_table:
        print(f"\n{'Iteration':<10} {'x_n':<15} {'g(x_n)':<15} {'Error':<15} {'f(x_n)':<15}")
        print("-"*70)
        print(f"{0:<10} {x:<15.8f} {'-':<15} {'-':<15} {f(x):<15.8f}")
    
    for i in range(1, max_iter + 1):
        x_new = g(x)
        error = abs(x_new - x)
        
        history.append([i, x_new, f(x_new)])
        
        if print_table:
            print(f"{i:<10} {x_new:<15.8f} {g(x):<15.8f} {error:<15.8f} {f(x_new):<15.8f}")
        
        if error < tol:
            if print_table:
                print(f"\n✓ Converged to root: {x_new:.10f}")
                print(f"  Iterations: {i}")
                print(f"  f(root) = {f(x_new):.2e}")
            return x_new, i, True, history
        
        x = x_new
    
    if print_table:
        print(f"\n⚠ Did not converge within {max_iter} iterations")
    
    return x, max_iter, False, history

def check_convergence(g_prime, x0):
    """
    Check if fixed point iteration will converge based on |g'(x0)| ≤ 1
    
    Parameters:
    g_prime: function - derivative of g(x)
    x0: float - initial guess
    
    Returns:
    bool: True if condition satisfied, False otherwise
    float: value of |g'(x0)|
    """
    try:
        g_prime_x0 = abs(g_prime(x0))
        condition_satisfied = g_prime_x0 <= 1
        return condition_satisfied, g_prime_x0
    except:
        return None, None