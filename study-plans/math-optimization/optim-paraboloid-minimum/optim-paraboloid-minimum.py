def paraboloid_minimum(a, b, c, d, e):
    """
    Returns: dict with 'x_star', 'y_star', 'f_min' (floats), each rounded to 6 decimals
    """
    
    x_star = np.round(-c/(2.0*a),6)
    y_star = np.round(-d/(2.0*b),6)
    f_min = np.round(e - (c*c/(4.0*a)) - (d*d/(4.0*b)),6)

    return {"x_star":x_star, "y_star":y_star, "f_min":f_min}