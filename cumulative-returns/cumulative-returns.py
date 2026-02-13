def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    import numpy as np 

    returns = np.asarray(returns)

    cum_re = np.zeros(len(returns))
    w = 1.0

    for i in range(len(returns)):
        w = w * (1+returns[i])
        cum_re[i] = w-1

    return cum_re.tolist()    
