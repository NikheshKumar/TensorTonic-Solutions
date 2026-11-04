import numpy as np

def gradient_check(f_name: str, x: float, analytical_deriv: float) -> dict:
    """
    Returns finite-difference errors and the best h for each method.
    """
    d = {"sin":np.sin, "exp":np.exp, "cos":np.cos, "sigmoid": lambda z: (1.0/(1.0+np.exp(-z))), "x_cubed": lambda z: z**3}
    f = d[f_name]

    fe, ce, be = [], [], []
    ad = analytical_deriv
    h_range = []


    for i in range(1, 9):
        h = 10.0 ** (-i)
        h_range.append(h)
        
        g_forward = (f(x+h)-f(x))/h
        g_backward = (f(x)-f(x-h))/h
        g_central = (f(x+h)-f(x-h))/(2.0*h)

        fe.append(float(abs(g_forward - ad)))
        ce.append(float(abs(g_central - ad)))
        be.append(float(abs(g_backward - ad)))

    best_h_f = float(h_range[int(np.argmin(fe))])
    best_h_c = float(h_range[int(np.argmin(ce))])
    best_h_b = float(h_range[int(np.argmin(be))])


    return {"h_values":h_range, "forward_errors":fe, "backward_errors":be, "central_errors":ce, "optimal_h_forward":best_h_f, "optimal_h_backward":best_h_b, "optimal_h_central":best_h_c}


    