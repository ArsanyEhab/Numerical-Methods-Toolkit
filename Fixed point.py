import numpy as np

def fixed_point_iteration(f, g, x0, tol=1e-6, max_iter=100, print_table=True):
  
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
   
    try:
        g_prime_x0 = abs(g_prime(x0))
        condition_satisfied = g_prime_x0 <= 1
        return condition_satisfied, g_prime_x0
    except:
        return None, None
