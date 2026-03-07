def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    import numpy as np 

    x = np.float64(x0)

    for i in range(steps):
      grad = 2*a*x + b
      x = x - lr * grad

    return float(x)