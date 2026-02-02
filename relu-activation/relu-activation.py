import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.atleast_1d(np.asarray(x, dtype=float))

    y = np.maximum(0.0, x)

    return y