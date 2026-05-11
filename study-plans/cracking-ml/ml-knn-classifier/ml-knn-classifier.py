import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    import numpy as np 

    X_train = np.asarray(X_train, np.float64)
    y_train = np.asarray(y_train, int)
    X_test = np.asarray(X_test, np.float64)

    sq_test = np.sum(X_test**2, axis=1, keepdims=True)
    sq_train = np.sum(X_train**2, axis=1, keepdims=True).T

    dist = np.sqrt(np.maximum(0, sq_test + sq_train - 2 * np.dot(X_test, X_train.T)))

    nearest_idx = np.argsort(dist, axis=1)[:, :k]

    nearest_labels = y_train[nearest_idx]

    y_pred = []
    for labels in nearest_labels:
        counts = np.bincount(labels)
        y_pred.append(int(np.argmax(counts)))


    return y_pred
    
    
