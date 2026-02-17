def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    import numpy as np 
    points = np.asarray(points)
    assignments = np.asarray(assignments)

    cen = np.zeros((k, points.shape[1]))
    count = np.zeros(k, float)

    for point, ass in zip(points, assignments):
      cen[ass] += point
      count[ass] += 1

    res = np.divide(cen, count[:, None], out=np.zeros_like(cen), where=count[:, None] != 0)
  
    return res.tolist()

      
      
      
    