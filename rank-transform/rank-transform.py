def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)

    sorted_val = np.sort(values)
    rank = np.zeros(len(values))

    for i in range(len(values)):
      indices = np.where(sorted_val == values[i])[0] + 1
      rank[i] = np.mean(indices)

    return rank.tolist()  

    
    