def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)

    d = {ordering[i]:i for i in range(len(ordering))}

    ans = [d[v] for v in values]

    return ans

    