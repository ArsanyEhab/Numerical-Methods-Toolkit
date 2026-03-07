import numpy as np

# Halley Method Function 
def halley_method(f, df, ddf, x0, tol=1e-6, max_iter=100, print_table=True):
    x = x0
    xs = [x]

    if print_table:
        print(f"\n{'Iteration':<10} {'x_n':<15} {'f(x_n)':<15} {'Error':<15}")
        print("-" * 55)
        try:
            fx0 = f(x)
        except Exception:
            fx0 = float("nan")
        print(f"{0:<10} {x:<15.8f} {fx0:<15.8f} {'-':<15}")

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)

        denominator = 2*(dfx**2) - fx*ddfx
        if denominator == 0:
            print("Division by zero encountered.")
            return xs

        x_new = x - (2*fx*dfx)/denominator
        xs.append(x_new)

        error = abs(x_new - x)
        if print_table:
            print(f"{i:<10} {x_new:<15.8f} {f(x_new):<15.8f} {error:<15.8f}")
        if error < tol:
            print("Root =", x_new)
            print("Iterations =", i)
            return xs

        x = x_new

    print("Did not converge within max iterations.")
    return xs

def main():
    # User Input
    print("Enter function in terms of x (use ** for powers, np for math functions)")
    f_input = input("f(x) = ")
    df_input = input("f'(x) = ")
    ddf_input = input("f''(x) = ")
    x0 = float(input("Initial guess x0 = "))

    # Convert input to Python functions
    f = lambda x: eval(f_input)
    df = lambda x: eval(df_input)
    ddf = lambda x: eval(ddf_input)

    # Run Halley method
    halley_method(f, df, ddf, x0, print_table=True)


if __name__ == "__main__":
    main()