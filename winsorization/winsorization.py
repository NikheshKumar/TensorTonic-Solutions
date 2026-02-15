def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)
    w = np.sort(values)
    n = len(values)
    if n == 0:
        return []
  
    low = np.percentile(w, lower_pct)
    high = np.percentile(w, upper_pct)
  
    w = np.clip(w, low, high)

    return w.tolist()

    