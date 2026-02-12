import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    # Write code here

    AP = []

    for y_true, y_score in zip(y_true_list, y_score_list):

        y_true = np.asarray(y_true, float)
        y_score = np.asarray(y_score, float)

        R = np.sum(y_true)

        if R == 0:
            AP.append(0.0)
            continue

        indices = np.argsort(y_score)[::-1]
        if k is not None:
            indices = indices[:k]

        y_true_sorted = y_true[indices]  

        rel = np.cumsum(y_true_sorted)

        ranks = [i for i in range(1, len(y_true_sorted) + 1) ]

        p = rel / ranks

        val = np.sum(p * y_true_sorted) / R
        AP.append(val)  


    mAP = np.mean(AP)

    return [mAP, AP]