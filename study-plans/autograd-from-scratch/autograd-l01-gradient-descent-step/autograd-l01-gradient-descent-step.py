import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    
    values_new = values - learning_rate * gradients

    l = np.sum(gradients * (values_new - values))

    return ([float(v) for v in values_new], l)
    
