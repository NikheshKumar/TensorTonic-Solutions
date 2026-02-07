def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # Write code here

    import numpy as np 

    grad = np.asarray(grad, float)
    m = len(s_list)
    s_list = np.asarray(s_list, float)
    y_list = np.asarray(y_list, float)

    alpha = np.zeros(m, float)
    rho = np.zeros(m, float)

    if m == 0:
        return -grad

    q = grad.copy()

    for i in range(m-1, -1, -1):
        rho[i] = 1.0 / np.dot(y_list[i].T,s_list[i])
        alpha[i] = rho[i] * np.dot(s_list[i], q)
        q = q - alpha[i] * y_list[i]

    num = np.dot(s_list[-1], y_list[-1])
    den = np.dot(y_list[-1].T, y_list[-1])

    if den == 0:
        gamma = 1.0
    else:
        gamma = num / den

    r = gamma * q    

    for i in range(m):
        beta = rho[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha[i] - beta)


    direction = -r
    return direction.tolist()        