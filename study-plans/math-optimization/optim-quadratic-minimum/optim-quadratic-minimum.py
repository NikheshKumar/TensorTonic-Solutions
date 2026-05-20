def quadratic_minimum(a, b, c):
    """
    Returns: dict with 'x_star' and 'f_min' (floats), each rounded to 6 decimals
    """
    import numpy as np 

    x_star = -b/ (2*a)

    f_min = a*(x_star**2) + b*x_star + c

    return {"x_star":round(float(x_star),6), "f_min":round(float(f_min),6)}
