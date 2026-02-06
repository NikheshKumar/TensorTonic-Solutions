def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here

    import numpy as np 

    predictions = np.asarray(predictions, float)
    K = predictions.shape[0]

    q = np.full_like(predictions, fill_value=epsilon / K)
    q[target] = (1.0 - epsilon) + (epsilon / K)

    e = 1e-12
    p = np.clip(predictions, e, 1.0)

    L = - np.sum(q * np.log(p))

    return float(L)