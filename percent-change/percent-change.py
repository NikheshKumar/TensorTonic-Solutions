def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    import numpy as np
    series = np.asarray(series, dtype=float)
    ans = []

    for i in range(1,len(series)):

        if series[i-1] != 0.0:
            ans.append( (series[i] - series[i-1]) / series[i-1] )
        if series[i-1]==0.0:
            ans.append(0.0)    

    return ans        
