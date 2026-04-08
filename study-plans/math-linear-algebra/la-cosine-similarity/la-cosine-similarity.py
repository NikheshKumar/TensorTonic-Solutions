import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    norm_a = np.linalg.norm(a, ord=2)
    norm_b = np.linalg.norm(b, ord=2)

    if abs(np.minimum(norm_a, norm_b)) < 1e-8:
        return 0.000000

    return np.dot(a, b) / (norm_a*norm_b)