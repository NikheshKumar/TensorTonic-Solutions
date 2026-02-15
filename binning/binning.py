def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    import numpy as np 
    values = np.asarray(values)
  
    if values.size==0:
      return []  

    w = ( np.max(values) - np.min(values) ) / num_bins

    if w == 0:
      return [0]*len(values)

    bins = (values - np.min(values) ) // w
    bins = np.minimum(bins, num_bins - 1).astype(int)

    return bins.tolist()