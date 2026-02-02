import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    m,n = A.shape
    B = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            B[i][j] = A[j][i]

    return B        
