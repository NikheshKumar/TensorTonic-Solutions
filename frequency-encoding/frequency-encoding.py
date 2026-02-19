def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    import numpy as np 
    from collections import Counter

    values = np.asarray(values)
    count = Counter(values)
    ans = np.zeros(len(values))
    

    for i in range(len(values)):
      prob = count[values[i]] / len(values)
      ans[i] = prob

    return ans.tolist()