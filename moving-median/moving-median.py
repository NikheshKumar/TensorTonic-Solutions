def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)
    n = len(values)

    n_windows = n - window_size + 1

    ans = []

    for i in range(n_windows):
        window = values[i : i + window_size]
        med = float(np.median(window))
        ans.append(med)

    return ans            
