import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors = np.asarray(vectors, np.float64)
    coefficients = np.asarray(coefficients, np.float64)

    return coefficients @ vectors