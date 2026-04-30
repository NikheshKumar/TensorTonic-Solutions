def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    import numpy as np 

    x = np.asarray(series, np.float64)
    n = len(x)

    if n == 0:
        return []

    if np.var(x) == 0.0:
        res = [0.0] * (max_lag + 1)
        res[0] = 1.0
        return res

    res = []

    for i in range(max_lag+1):

        x_mean = np.mean(x)
        gamma_0 = np.var(x)

        gamma_k = np.sum( (x[:n-i] - x_mean)*(x[i:] - x_mean) ) / n

        res.append((gamma_k/gamma_0).astype(np.float64))

    return res