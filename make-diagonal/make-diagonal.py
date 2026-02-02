import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.asarray(v, dtype=float)
    n = len(v)

    D = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i==j:
                D[i][j] = v[i]

    return D            
