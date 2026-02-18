import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y, float)

    classes, counts = np.unique(y, return_counts=True)

    p = counts / len(y)
    p = p[p>0.0]
    H = - np.sum(p*np.log2(p))

    return H