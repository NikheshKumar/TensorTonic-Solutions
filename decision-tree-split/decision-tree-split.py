import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n_samples, n_features = X.shape

    def gini_calculation(labels):
      if len(labels)==0:
        return 0.0
      cla, count = np.unique(labels, return_counts=True)
      p = count / len(labels)
      return 1.0 - np.sum(p**2)

    gini_parent = gini_calculation(y)
    best_info_gain = -np.inf
    best_feature = None
    best_threshold = None

    eps = 1e-8
      

    for i in range(n_features):
      sorted_indices = np.argsort(X[:, i])
      sorted_X = X[sorted_indices, i]
      sorted_y = y[sorted_indices]

      for s in range(1,n_samples):
        threshold = (sorted_X[s] + sorted_X[s-1]) / 2.0

        y_left = sorted_y[:s]
        y_right = sorted_y[s:]

        p_left = len(y_left) / n_samples
        p_right = 1.0 - p_left

        gini_split =  p_left * gini_calculation(y_left) + p_right * gini_calculation(y_right)
        gain = gini_parent - gini_split

        if gain > best_info_gain + eps:
          best_info_gain = gain 
          best_feature = i
          best_threshold = threshold
      

      
    return [best_feature, best_threshold]
      