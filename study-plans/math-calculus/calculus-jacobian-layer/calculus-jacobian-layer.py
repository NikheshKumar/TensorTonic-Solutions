import numpy as np

def jacobian_network_layer(W: list, b: list, x: list) -> dict:
    """
    Returns layer values and analytical and numerical Jacobians.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    def _sigmoid(y):
        return np.exp(y)/(1.0 + np.exp(y))

    z = W @ x + b

    f = _sigmoid(z)

    J_ana = _sigmoid(z)[:, None]*(1.0 - _sigmoid(z)[:, None])*W

    d = x.shape[0]
    m = f.shape[0]

    J_num = np.zeros((m,d), dtype=np.float64)
    eps = 1e-5

    for i in range(0,d):
        x_plus = x.copy()
        x_plus[i] += eps 
        x_minus = x.copy()
        x_minus[i] -= eps 

        J_num[:, i] = (_sigmoid(W@x_plus + b) - _sigmoid(W@x_minus + b)) / (2.0*eps)

    def convert(z):
        return np.asarray(z).tolist()

    return {"z":convert(z), "f":convert(f), "analytical_jacobian":convert(J_ana), "numerical_jacobian":convert(J_num)}