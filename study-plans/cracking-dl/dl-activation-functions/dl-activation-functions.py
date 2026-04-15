import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = np.asarray(x, np.float64)

    if activation=="relu":
        out = np.maximum(x,0.0)
        derivative = (x>0).astype(np.float64)
        
    if activation=="leaky_relu":
        out = np.maximum(x, 0.01 * x)
        derivative = np.where(x > 0, 1.0, 0.01)
        
    if activation=="sigmoid":
        out = 1 / (1 + np.exp(-x))
        derivative = out * (1 - out)
        
    if activation=="tanh":
        out = np.tanh(x)
        derivative = 1 - out**2
        
    if activation=="gelu":
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        out = 0.5 * x * (1 + tanh_inner)
        derivative = (0.5 * (1 + tanh_inner) + (0.5 * x * (1 - tanh_inner**2) * np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x**2))).astype(np.float64)
        
    if activation=="swish":
        sigmoid_x = 1 / (1 + np.exp(-x))
        out = x * sigmoid_x
        derivative = out + sigmoid_x * (1 - out)

    return [np.round(out,4), np.round(derivative,4)]

        
    
