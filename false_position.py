def false_position(func, a, b, tol=1e-6, max_iter=100):
    """
    Finds the root of a function using the False Position method.
    """
    # Check if the root is even in the interval
    if func(a) * func(b) >= 0:
        print("False Position method fails. f(a) and f(b) must have opposite signs.")
        return None

    print(f"{'Iter':<10} {'c':<15} {'f(c)':<15}")
    print("-" * 40)

    for i in range(max_iter):
        # Calculate the point c using the False Position formula
        # c = [a*f(b) - b*f(a)] / [f(b) - f(a)]
        fa, fb = func(a), func(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = func(c)

        print(f"{i+1:<10} {c:<15.6f} {fc:<15.6e}")

        # Check if we've reached the desired tolerance
        if abs(fc) < tol:
            return c

        # Decide which side to keep
        if func(a) * fc < 0:
            b = c
        else:
            a = c

    print("Reached maximum iterations.")
    return c

# --- Example Usage ---
if __name__ == "__main__":
    # Define the function: f(x) = x^3 - x - 2
    my_function = lambda x: x**3 - x - 2

    # Initial guesses where f(a) is negative and f(b) is positive
    lower_bound = 1
    upper_bound = 2

    root = false_position(my_function, lower_bound, upper_bound)
    
    if root:
        print("-" * 40)
        print(f"Final Root: {root:.6f}")