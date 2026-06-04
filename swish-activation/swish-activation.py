import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    num = x * np.exp(x)
    den = 1 + np.exp(x)

    return num/den