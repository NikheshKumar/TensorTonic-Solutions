def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    def df(a,b,x):
        return 2*a*x + b

    x = x0
    for i in range(steps):
        x = x - lr * df(a,b,x)

    return x    
