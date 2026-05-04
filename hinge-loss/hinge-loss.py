import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_score = np.asarray(y_score, np.float64)
    y_true = np.asarray(y_true, np.float64)
        
    loss = np.maximum(0, margin - y_score*y_true)
    ans = np.mean(loss) if reduction=="mean" else np.sum(loss)

    return np.float64(ans)

        

        