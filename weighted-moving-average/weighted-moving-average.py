def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)
    weights = np.asarray(weights)

    n = len(values)
    k = len(weights)

    wma = np.zeros(n-k+1)

    for i in range(n-k+1):

        num = np.sum( weights[0:k]*values[i:i+k] )
        den = np.sum( weights[0:k] )
        wma[i] = num / den

    return wma.tolist()   