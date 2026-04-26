import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X = np.asarray(X)
    labels = np.asarray(labels)

    n_samples, n_features = X.shape

    dist_mat = X[:,None,:] - X[None,:,:]

    euclidean_dist = np.linalg.norm(dist_mat, axis=2)

    sil_mat = np.zeros((n_samples,))

    for i in range(n_samples):

        same = labels==labels[i]
        other = ~same

        same[i] = False

        if np.any(same) :
            a = np.mean(euclidean_dist[i, same])
        else:
            a = 0.0

        b = float('inf')

        for l in np.unique(labels[other]):
            mask = labels==l
            b = np.minimum(b, np.mean(euclidean_dist[i, mask]))

        sil_mat[i] = (b-a)/(np.maximum(a,b))


    return np.mean(sil_mat).astype(np.float64)

    