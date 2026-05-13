def reinforce_loss(log_probs, returns):
    """
    Returns: float, REINFORCE policy loss rounded to 4 decimals
    """
    import numpy as np 

    log_probs = np.asarray(log_probs, np.float64)
    returns = np.asarray(returns, np.float64)

    l = -np.mean(log_probs*returns)

    return round(float(l),4)

    
