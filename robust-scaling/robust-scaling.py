def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values, float)

    values_sorted = np.sort(values)
    n = len(values)

    def calculate_median(arr):
      n = len(arr)
      if n==0:
        return 0.0
      if n%2==0:
        median = (arr[n//2] + arr[n//2 - 1]) / 2.0
      else:
        median = arr[n//2]
      return median

  
    q1 = calculate_median(values_sorted[:n//2])
    q3 = calculate_median(values_sorted[n//2:]) if n%2==0 else calculate_median(values_sorted[n//2 + 1:])
    med = calculate_median(values_sorted)

    x_scaled = (values - med) / (q3 - q1) if q3-q1!=0 else (values - med)

    return x_scaled.tolist()