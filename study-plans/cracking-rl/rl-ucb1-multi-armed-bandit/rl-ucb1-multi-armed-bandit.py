import math 

def ucb1_scores(Q, N, t, c):
    """
    Returns: list of K UCB1 scores, each rounded to 4 decimals
    """
    import numpy as np 

    Q = np.asarray(Q, np.float64)
    N = np.asarray(N, np.float64)

    scores = Q + c * np.sqrt(np.log(t)/N)

    return [round(s,4) for s in scores]
