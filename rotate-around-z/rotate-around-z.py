import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Your code here
    points = np.asarray(points)

    row1 = [np.cos(theta), -np.sin(theta), 0.0]
    row2 = [np.sin(theta), np.cos(theta), 0.0]
    row3 = [0.0,0.0,1.0]

    rot_mat = np.array([row1, row2, row3])

    q = points @ rot_mat.T

    return q    