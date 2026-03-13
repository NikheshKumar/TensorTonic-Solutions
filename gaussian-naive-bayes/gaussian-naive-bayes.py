def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    import numpy as np  

    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    y_train = np.asarray(y_train, float)

    cla, counts = np.unique(y_train, return_counts=True)

    n_classes = len(cla)
    n_features = X_train.shape[1]

    priors = counts / len(y_train)

    mu = np.zeros((n_classes, n_features), float)
    var = np.zeros((n_classes, n_features), float)

    eps = 1e-9

    for i, c in enumerate(cla):
        X_class = X_train[y_train == c]
        mu[i, :] = np.mean(X_class, axis=0)
        var[i, :] = np.var(X_class, axis=0) + eps

    X_test_new = X_test[:, np.newaxis, :]

    log_posterior = np.log(priors) + np.sum( np.log(2 * np.pi * var) - 0.5 * ((X_test_new- mu)**2 / var), axis=2)

    y_pred_index = np.argmax(log_posterior, axis=1)
    y_pred = cla[y_pred_index]

    return y_pred.tolist()