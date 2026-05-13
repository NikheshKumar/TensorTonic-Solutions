def softmax_action_probs(Q_values, tau):
    """
    Returns: list of length A, action probabilities under softmax/Boltzmann, rounded to 4 decimals
    """
    import numpy as np 

    Q_values = np.asarray(Q_values, np.float64)

    Q = Q_values - np.max(Q_values)

    num = np.exp(Q/tau)
    den = np.sum(np.exp(Q/tau))

    return [round(p,4) for p in num/den]
