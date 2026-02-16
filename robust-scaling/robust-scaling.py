def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    import numpy as np 

    values = np.asarray(values, float)
    values_sorted = np.sort(values)
    n = len(values)

    def compute_median(arr):
      n = len(arr)
      if n==0:
        return 0.0  
      if n%2==0:
        med = (arr[(n//2)] + arr[(n//2)-1]) / 2.0
      else:
        med = arr[(n//2)]
      return med

    if n % 2 == 1:
        q1 = compute_median(values_sorted[:n//2])
        q3 = compute_median(values_sorted[n//2 + 1:])
    else:
        q1 = compute_median(values_sorted[:n//2])
        q3 = compute_median(values_sorted[n//2:])
      
    med = compute_median(values_sorted)
    
    if q3-q1 == 0:
      return (values-med).tolist()

    val_scaled = ( values - med ) / (q3 - q1)

    return val_scaled.tolist()
    