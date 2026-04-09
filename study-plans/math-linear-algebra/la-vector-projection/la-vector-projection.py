import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)

    proj = (np.dot(u,v) / np.dot(v,v)) * v if np.dot(v,v)>0.0 else 0.0

    return proj