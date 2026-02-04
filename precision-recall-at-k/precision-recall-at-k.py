def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here

    import numpy as np 

    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)

    top_k = recommended[:k]
    t = set(top_k)

    precision = len(t & set(relevant) ) / k if k>0 else 0.0
    recall = len(t & set(relevant) ) / len(relevant) if len(relevant)>0 else 0.0

    return [precision, recall]