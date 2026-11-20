import numpy as np

def activation_continuity_analysis(x: list) -> dict:
    """
    Returns nondifferentiable tested points for three activations.
    """
    x = np.asarray(x, dtype=np.float64)

    def relu(z):
        return np.maximum(0.0, z)

    def leaky_relu(z):
        return np.maximum(alpha * z, z) 

    def gelu(z):
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (z +0.044715*(z**3))))

    functions = {"relu":relu, "gelu":gelu, "leaky_relu":leaky_relu}

    res = {}
    h = 0.0000001
    alpha = 0.01
    
    for name, f in functions.items():
        left = (f(x) - f(x-h)) / h 
        right = (f(x+h) - f(x)) / h 
        diff = left - right
        mask = np.abs(diff) > 1e-5
        res[name] = x[mask].tolist()

    return res

