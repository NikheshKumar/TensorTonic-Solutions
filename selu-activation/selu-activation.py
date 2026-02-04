def selu(x):
    """
    Apply SELU activation to each element.
    """
    # Write code here
    import numpy as np 

    x = np.asarray(x, float)

    lam = 1.0507
    alpha = 1.6733
    
    y = np.exp(x) - 1

    ans = np.where(x>0.0, lam*x, lam*alpha*y)

    return ans.tolist()