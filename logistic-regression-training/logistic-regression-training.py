import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, float)
    y = np.asarray(y, float)
  
    w = np.zeros(len(X[0]))
    b = 0.0
    
    for i in range(steps):
      
      z = np.dot(X,w) + b
      p = _sigmoid(z) 
    
      
      gradw = np.mean(X.T@(p-y))
      gradb = np.mean(p-y)
  
      w = w - lr * gradw
      b = b - lr * gradb

    return w,b