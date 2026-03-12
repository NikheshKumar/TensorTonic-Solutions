def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    import numpy as np 

    points = np.asarray(points)
    assignments = np.asarray(assignments)

    centroid = np.zeros((k,points.shape[1]))

    count = np.zeros(k)

    for p, a in zip(points, assignments):
      centroid[a] += p
      count[a] += 1

    ans = np.divide(centroid, count[:,None], out=np.zeros_like(centroid), where=count[:, None] != 0 )

    return ans.tolist()