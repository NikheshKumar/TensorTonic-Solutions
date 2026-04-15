import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    num_samples, num_features = X.shape

    w = np.zeros(num_features)
    b = 0.0

    for j in range(epochs):
        for i in range(num_samples):
        
            z = np.dot(w, X[i]) + b
            
            y_hat = 1.0 if z >= 0 else 0.0

            w = w + lr * (y[i] - y_hat) * X[i]
            b = b + lr * (y[i] - y_hat)

    return w, b    