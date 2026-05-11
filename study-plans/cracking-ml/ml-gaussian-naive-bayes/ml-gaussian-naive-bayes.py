import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    import numpy as np 

    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=np.float64)

    eps = 1e-9

    cla = np.unique(y_train)
    n_features = X_train.shape[1]
    n_classes = len(cla)

    mu = np.zeros((n_classes, n_features), np.float64)
    v = np.zeros((n_classes, n_features), np.float64)
    priors = np.zeros((n_classes,),np.float64)

    for i, c in enumerate(cla):
        X_c = X_train[y_train == c]
        mu[i, :] = np.mean(X_c, axis=0)
        v[i, :] = np.var(X_c, axis=0) + eps
        priors[i] = X_c.shape[0] / X_train.shape[0]

        
    y_pred = []

    for x in X_test:
        posteriors = []
        for i, c in enumerate(cla):
            pc = np.log(priors[i])
            likelihood = np.sum( -((x - mu[i, :]) ** 2) / (2 * v[i, :]) - np.log(np.sqrt(2 * np.pi * v[i, :])))
            posteriors.append(pc + likelihood)
        
        y_pred.append(int(cla[np.argmax(posteriors)]))

    return y_pred

    