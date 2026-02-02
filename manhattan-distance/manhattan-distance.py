import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x, y = np.asarray(x,float) , np.asarray(y, float)

    l1_dist = np.sum(np.abs(x-y))

    return l1_dist