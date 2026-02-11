import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here  

    if len(matrix) == 0:
        return None 

    if not isinstance(matrix, (list, tuple)):
        return None    

    if not all(isinstance(row, (list, tuple)) for row in matrix):
        return None    

    if any(len(row) != len(matrix[0]) for row in matrix):
        return None
    
    if len(matrix) != len(matrix[0]):
        return None       

    matrix = np.asarray(matrix, dtype=float)
    n, d = matrix.shape

    if matrix.ndim != 2:
        return None


    eig = np.linalg.eigvals(matrix)
    sorted_eig = np.sort(eig)

    return sorted_eig 