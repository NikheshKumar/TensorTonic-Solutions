import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.atleast_1d(np.array(x, np.float64))
    x = np.clip(x, -100, 100)
    
    num = x * np.exp(x)
    den = 1 + np.exp(x)

    return num/den