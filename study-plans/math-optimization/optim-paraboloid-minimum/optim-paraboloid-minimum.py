def paraboloid_minimum(a, b, c, d, e):
    """
    Returns: dict with 'x_star', 'y_star', 'f_min' (floats), each rounded to 6 decimals
    """
    import numpy as np

    x_star = -c / (2*a)

    y_star = -d / (2*b)

    f_min = a*(x_star**2) + b*(y_star**2) + c*x_star + d*y_star + e


    return {'x_star':round(float(x_star),6), 'y_star':round(float(y_star),6), 'f_min':round(float(f_min),6) }
