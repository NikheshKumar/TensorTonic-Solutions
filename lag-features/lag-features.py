def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    # Write code here
    import numpy as np 

    series = np.asarray(series)
    max_lag = np.max(lags)
    k = len(series)
    ans = []

    for i in range(max_lag, k, 1):
      temp = []
      for lag in lags:
        window = series[i-lag]
        temp.append(window)
      ans.append(temp)

    return ans
    