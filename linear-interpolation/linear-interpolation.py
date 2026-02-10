def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here

    import numpy as np 

    values = np.asarray(values, float)

    ans = np.array([np.nan if v==None else v for v in values], dtype=float) 

    val = ~np.isnan(ans)

    tot = np.arange(ans.shape[0])

    if val is None:
        return val

    ans[np.isnan(ans)] = np.interp(tot[np.isnan(ans)], tot[val], ans[val])

    return ans.tolist()    


    