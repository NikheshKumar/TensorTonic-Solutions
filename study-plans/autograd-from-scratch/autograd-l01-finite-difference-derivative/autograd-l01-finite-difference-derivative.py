import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    
    f_x = np.sum([coefficients[k] * x**k for k in range(len(coefficients))])
    f_xh = np.sum([coefficients[k] * (x+h)**k for k in range(len(coefficients))])
    slope = (f_xh - f_x) / h

    return f_x, f_xh, slope
