import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    vectors = np.asarray(vectors, np.float64)
    k, n = vectors.shape

    res = []

    for i in range(k):
        v = vectors[i]
        u = v.copy()
        for q in res:
            u -= np.dot(v, q) * q
        u = u / np.linalg.norm(u)
        res.append(u)

    return res
        
        