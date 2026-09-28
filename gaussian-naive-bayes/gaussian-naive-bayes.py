import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    
    classes = set(y_train)
    sorted_classes = sorted(classes)
    N, D = len(X_train), len(X_train[0])
    eps = 1e-9

    priors = {}
    means = {}
    var = {}

    for c in classes:
        X_c = [X_train[i] for i in range(N) if y_train[i] == c]
        priors[c] = len(X_c) / N
        means[c]  = [sum(X_c[i][j] for i in range(len(X_c))) / len(X_c) for j in range(D)]
        var[c]    = [sum((X_c[i][j] - means[c][j]) ** 2.0 for i in range(len(X_c))) / len(X_c) + eps for j in range(D)]

    pred = [0] * len(X_test)

    for i in range(len(X_test)):
        
        best_c = None
        best_score = -float("inf")
        
        for c in classes:
            score = math.log(priors[c])
            for j in range(D):
                mu    = means[c][j]
                v = var[c][j]
                score += -0.5 * math.log(2.0 * math.pi * v) - (X_test[i][j] - mu) ** 2.0 / (2.0 * v)
                         
            if score > best_score:
                best_score = score
                best_c = c

        pred[i] = best_c

    return pred

    
        
        