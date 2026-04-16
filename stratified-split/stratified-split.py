import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    
    X = np.asarray(X)
    y = np.asarray(y)

    if rng is None:
        rng = np.random

    cla, counts = np.unique(y, return_counts=True)

    train_indices = []
    test_indices = []

    for c in cla:


        class_indices = np.where(y == c)[0]
        rng.shuffle(class_indices)  

        n_test = max(1, int(len(class_indices) * test_size))

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    train_indices = np.array(sorted(train_indices), dtype=int)
    test_indices = np.array(sorted(test_indices), dtype=int)
    

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    