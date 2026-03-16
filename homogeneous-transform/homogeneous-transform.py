import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T, float)
    points = np.asarray(points, float)

    n = points.ndim

    points_new = np.atleast_2d(points)
  
    ones = np.ones((points_new.shape[0], 1))
    points_h = np.hstack([points_new, ones])

    p_t = points_h @ T.T 

    q = p_t[:, :3]

    w = p_t[:,3:4]

    mask = (np.abs(w) > 1e-9).flatten()

    res = np.zeros_like(q, float)

    res[mask] = q[mask] / w[mask]
    res[~mask] = q[~mask]

    if n==1:
      return res[0]
 
    return res
  
    
  