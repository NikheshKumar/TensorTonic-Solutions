import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    tr = sum( A[i][i] for i in range(len(A)) )

    return tr
