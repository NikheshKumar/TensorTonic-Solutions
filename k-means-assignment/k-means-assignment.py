def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    import numpy as np 

    points = np.asarray(points)
    centroids = np.asarray(centroids)
  
    diff = np.sum( (centroids-points[:,np.newaxis])**2, axis=2 )

    ans = np.argmin(diff, axis=1)

    return ans.tolist()
    