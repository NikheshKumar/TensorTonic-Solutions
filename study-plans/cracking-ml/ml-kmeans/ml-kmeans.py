import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """

    X = np.array(X, dtype=np.float64)
    N, D = X.shape
    
    rng = np.random.RandomState(seed)

    idx = rng.choice(N, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(N, dtype=int)

    for _ in range(max_iters):

        dist = np.sum(((X[:, None, :] - centroids[None, :, :]) ** 2), axis=2)
        new_labels = np.argmin(dist, axis=1)

        new_centroids = centroids.copy()
        
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centroids[j] = np.mean(X[mask], axis=0)


        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            labels = new_labels
            break

        centroids = new_centroids
        labels = new_labels

    return labels.tolist(), np.round(centroids, 4).tolist()

    
