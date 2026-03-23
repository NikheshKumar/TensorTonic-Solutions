def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    import numpy as np 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    tp = fp = fn = 0.0
    
    for c in classes:
        tp += np.sum((y_true==c) & (y_pred==c)) 
        fp += np.sum((y_true!=c) & (y_pred==c)) 
        fn += np.sum((y_true==c) & (y_pred!=c)) 


    f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn)>0 else 0.0

    return f1