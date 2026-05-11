import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    """
    Returns: A list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=np.float64)

    w = np.zeros((X_train.shape[1],), np.float64)
    b = 0.0
    w_sum = np.zeros((X_train.shape[1],), np.float64)
    b_sum = 0.0
    tot = 0


    y_pred = []

    for i in range(n_epochs):
        for j in range(X_train.shape[0]):
            
            if y_train[j] * (w @ X_train[j] + b) <= 0:
                w = w + y_train[j] * X_train[j]
                b = b + y_train[j]

            w_sum += w
            b_sum += b
            tot += 1
        

    avg_w = w_sum / tot
    avg_b = b_sum / tot

    scores = np.dot(X_test, avg_w) + avg_b
    y_pred = np.where(scores > 0, 1, -1).tolist()
        

    return y_pred
