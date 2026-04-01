import math
import numpy as np

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    
    if k==0:
        return 0.0
        
    relevance_scores = np.asarray(relevance_scores, dtype=float)

    def compute_dcg(scores):
        num = 2.0**(scores) - 1
        den = np.log2(np.arange(2, len(scores) + 2))
        return np.sum(num/den)

    
    dcg = compute_dcg(relevance_scores[:k])
    ideal_dcg = compute_dcg(np.sort(relevance_scores)[::-1][:k])

    normalised_dcg = dcg/ideal_dcg if ideal_dcg!=0.0 else 0.0

    return float(normalised_dcg)

    
    
    