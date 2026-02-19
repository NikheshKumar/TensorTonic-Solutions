def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values, float)
    p = window_size

    sma = np.zeros(len(values)-p+1)

    for i in range(len(sma)):
      window = values[i:i+p]
      sma[i] = np.mean(window)

    return sma.tolist()
  