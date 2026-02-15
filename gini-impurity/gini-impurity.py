import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    n_total = len(y_left) + len(y_right)

    if n_total == 0:
          return 0.0

    i, counts_left = np.unique(y_left, return_counts=True)
    j, counts_right = np.unique(y_right, return_counts=True)
  
    gini_left = 1 - np.sum( (counts_left / len(y_left))**2 )
    gini_right = 1 - np.sum( (counts_right / len(y_right))**2 )

    gini =  ( len(y_left)*gini_left + len(y_right)*gini_right ) /  n_total

    return gini