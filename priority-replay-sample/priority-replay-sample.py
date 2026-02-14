def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    import numpy as np 

    priorities = np.asarray(priorities, float)
    N = len(priorities)

    if alpha==0:
        powered_priorities = priorities

    powered_priorities = priorities**alpha

    sampling_prob = powered_priorities/ np.sum(powered_priorities)

    weights = (N*sampling_prob)**(-beta) 

    w_new = weights / np.max(weights) 

    return [sampling_prob.tolist(), w_new.tolist()]