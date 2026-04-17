import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    dtype = np.float64 if normalize != 'none' else int

    if y_pred.size==0 or y_true.size==0:
        if num_classes is None:
            return np.zeros((0,0), dtype)
        else:
            return np.zeros((num_classes, num_classes), int)

    n = y_pred.shape[0]

    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)


    cm = np.zeros((num_classes, num_classes), dtype=dtype)

    indices = y_true * num_classes + y_pred
    counts = np.bincount(indices, minlength=num_classes**2)
    cm = counts.reshape(num_classes, num_classes).astype(dtype)

    if normalize=="true":
        sum_rows = np.sum(cm, axis=1, keepdims=True)
        cm = np.divide(cm, sum_rows, out=np.zeros_like(cm), where=sum_rows != 0)

    if normalize=="pred":
        sum_cols = np.sum(cm, axis=0, keepdims=True)
        cm = np.divide(cm, sum_cols, out=np.zeros_like(cm), where=sum_cols != 0)

    if normalize=="all":
        total_sum = np.sum(cm)
        cm = cm / total_sum if total_sum>0.0 else cm
    
    
    return cm


    