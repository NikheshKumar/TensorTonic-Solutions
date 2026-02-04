def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    import numpy as np

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    TP = FP = FN = 0.0

    for cla in classes:
        TP += np.sum((y_true == cla) & (y_pred == cla))
        FP += np.sum((y_true != cla) & (y_pred == cla))
        FN += np.sum((y_true == cla) & (y_pred != cla))

    ans = 2 * TP / (2*TP + FP + FN)


    return ans if (2*TP + FP + FN) != 0 else 0.0