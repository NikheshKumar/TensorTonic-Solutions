import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)


    if rng is None:
        rng = np.random.shuffle()

    cla, counts = np.unique(y, return_counts=True)

    test_indices = []
    train_indices = []

    for c in cla:

        cla_indices = np.where(y==c)[0]
        rng.shuffle(cla_indices)

        n_test = np.maximum(1, int(np.floor(len(cla_indices) * test_size)))

        test_indices.extend(cla_indices[:n_test])
        train_indices.extend(cla_indices[n_test:])

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]