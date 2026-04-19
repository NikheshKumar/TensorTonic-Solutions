def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    # Write code here
    import numpy as np 

    series = np.asarray(series, np.float64)
    res = []

    for p in range(period):

        sub_series = series[p::period]

        if sub_series.size > 0:
            res.append(np.mean(sub_series))
        else:
            res.append(0.0)

    return res