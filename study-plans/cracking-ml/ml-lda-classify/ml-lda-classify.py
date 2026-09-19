import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)

    N, D = X_train.shape
    classes = np.unique(y_train)
    K = len(classes)
    
    means  = np.zeros((K, D), dtype=np.float64)
    p = np.zeros((K,), dtype=np.float64)
    
    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        means[i]  = X_c.mean(axis=0)
        p[i] = len(X_c) / N

    S = np.zeros((D,D), dtype=np.float64)
    for i, c in enumerate(classes):
        X_c = X_train[y_train==c]
        S += (X_c - means[i]).T @ (X_c - means[i])
    
    sigma = S/(N-K) + (1e-6 * np.eye(D))

    W =  np.linalg.solve(sigma, means.T)
    
    b = -0.5 * np.sum(means * W.T, axis=1) + np.log(p)

    scores = X_test @ W + b
    preds = np.argmax(scores, axis=1)

    return classes[preds].tolist()