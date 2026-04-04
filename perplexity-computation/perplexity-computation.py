def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    import numpy as np 

    prob_distributions = np.asarray(prob_distributions, float)
    actual_tokens = np.asarray(actual_tokens, int)
    
    n = len(actual_tokens)

    if n==0:
        return None

    p = prob_distributions[np.arange(n), actual_tokens]

    p = np.clip(p, 1e-8, 1.0)

    H = -np.mean(np.log(p))

    return float(np.exp(H))


