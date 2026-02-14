import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.asarray(v, float)
    w = np.asarray(w, float)

    norm_v = np.sqrt(np.sum(v**2))
    norm_w = np.sqrt(np.sum(w**2))

    if np.abs(norm_v) < 1e-10 or np.abs(norm_w) < 1e-10:
        return np.nan

    u = np.dot(v,w) / (norm_v * norm_w)
    u = np.clip(u, -1.0, 1.0)

    return np.arccos(u)   

