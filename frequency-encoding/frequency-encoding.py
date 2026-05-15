def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    import numpy as np 
    from collections import Counter

    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return []

    count = Counter(values)

    ans = [count[v]/ n for v in values]

    return ans

    