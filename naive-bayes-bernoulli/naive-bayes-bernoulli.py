import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, float)

    X_test_2d = np.atleast_2d(X_test)

    cla, counts = np.unique(y_train, return_counts=True)

    theta = []
    for c in cla:
        X_class = X_train[y_train == c]
        prob = (np.sum(X_class, axis=0) + 1) / (len(X_class) + 2)
        theta.append(prob)

    theta = np.array(theta)

    log_priors = np.log(counts / len(y_train))

    log_likelihood = (
        X_test_2d @ np.log(theta).T +
        (1 - X_test_2d) @ np.log(1 - theta).T
    )

    log_probs = log_likelihood + log_priors

    return log_probs
