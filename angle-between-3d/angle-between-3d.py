import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.asarray(v, float)
    w = np.asarray(w, float)

    eps = 1e-7
    norm_v = np.sqrt(np.sum(v**2))
    norm_w = np.sqrt(np.sum(w**2))

    if np.minimum(np.abs(norm_v), np.abs(norm_w)) < eps:
        return np.nan

    angle = np.dot(v, w)/(norm_v * norm_w)
    angle = np.clip(angle, -1.0, 1.0)
    angle = np.arccos(angle)

    return angle

    