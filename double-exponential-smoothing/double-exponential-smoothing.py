def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    # Write code here
    import numpy as np 

    series = np.asarray(series)

    l = np.zeros(len(series))

    b = series[1] - series[0]
    l[0] = series[0]

    for t in range(1, len(series)):
        l[t] = alpha*series[t] + (1-alpha)*( l[t-1] + b)
        b = beta*(l[t] - l[t-1]) + (1-beta)*(b)

    return l.tolist()