import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
  
    cla, cla_freq = np.unique(y_train, return_counts=True)

    val = cla[np.argmax(cla_freq)]

    res = np.full_like(X_test, val)

    return res
  
    