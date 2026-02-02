import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.atleast_1d(np.asarray(x, dtype=float))

    x_clipped = np.clip(x, -100, 100)
    sig = 1.0 / (1.0 + np.exp(-x_clipped) )

    return x * sig