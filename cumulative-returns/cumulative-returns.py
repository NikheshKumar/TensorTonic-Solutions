def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    import numpy as np 

    returns = np.asarray(returns)

    w = 1 + returns
    cum_grwoth = np.cumprod(w)

    cum_re = cum_grwoth - 1

    return cum_re.tolist()
    

  
