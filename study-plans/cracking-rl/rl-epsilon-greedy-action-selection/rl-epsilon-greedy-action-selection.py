def epsilon_greedy_probs(Q_values, epsilon):
    """
    Returns: list of length A, action probabilities under epsilon-greedy, rounded to 4 decimals
    """
    import numpy as np 

    Q_values = np.asarray(Q_values, np.float64)

    a = np.argmax(Q_values)

    p = np.full(len(Q_values), epsilon / len(Q_values), np.float64) 

    p[a] += (1.0 - epsilon)

    return [round(i,4) for i in p]
                         
