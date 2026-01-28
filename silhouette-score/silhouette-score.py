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

    difference_mat = X[:, None, :] - X[None, :, :]

    euclidean_dist = np.linalg.norm(difference_mat, axis=2)  

    s = np.zeros(n_samples)

    for i in range(n_samples):

        same_cluster = labels == labels[i]
        other_cluster = labels != labels[i]

        same_cluster[i] = False

        if np.any(same_cluster):
            a = np.mean(euclidean_dist[i, same_cluster])
        else:
            a = 0.0

        b = float('inf')
        
        for l in np.unique(labels[other_cluster]):
            mask = labels == l
            b = min(b, np.mean(euclidean_dist[i, mask]))    

        s[i] = (b - a) / ( max( a, b ) )

    return np.mean(s)