def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    import numpy as np 

    x = np.asarray(x,float)
    y = np.exp(x) - 1

    return np.where(x>0.0, x, alpha*y ).tolist()