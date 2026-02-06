def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here

    import numpy as np 
    
    series = np.asarray(series)

    if order == 0:
        return series.copy()

    if order >= len(series):
        return []    

    diff = series

    for _ in range(order):
        diff = [diff[i] - diff[i-1] for i in range(1, len(diff))]


    return diff        
