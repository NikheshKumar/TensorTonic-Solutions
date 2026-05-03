def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values, np.float64)
    n = len(values)
    k = window_size

    if k<0 or k>n:
        return []

    dev = np.zeros((n-k+1,), np.float64)

    for i in range(n-k+1):
        window = values[i:i+k]
        mu = np.mean(window)
        dev[i] = np.sqrt(np.mean((window - mu)**2))

    return dev.tolist()

    

        
    
        