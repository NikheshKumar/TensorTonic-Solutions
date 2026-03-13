import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x, float)

    sig = np.exp(x) / (1+np.exp(x))

    return x*sig