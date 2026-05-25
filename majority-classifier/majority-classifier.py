import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test)

    cla, counts = np.unique(y_train, return_counts=True)

    maj = cla[np.argmax(counts)]

    y_pred = np.full_like(X_test, maj, dtype=int)

    return y_pred
    