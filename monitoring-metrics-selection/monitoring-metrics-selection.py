def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here

    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if system_type == "classification":

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        N = TP + TN + FP + FN
        accuracy = (TP + TN) / N if N > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return [("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)]



    elif system_type == "regression": 
        mae = np.mean(np.abs(y_pred-y_true))
        rmse = np.sqrt( np.mean( (y_pred-y_true)**2 ) )

        return [("mae", mae),("rmse", rmse)] 


    elif system_type == "ranking":

        k=3
        arr = np.argsort(-y_pred)
        top_k = arr[:k]

        relevant_at_k = np.sum(y_true[top_k])
        N = np.sum(y_true)

        precision_at_3 = relevant_at_k / 3

        recall_at_3 = relevant_at_k / N if N > 0 else 0.0

        return [("precision_at_3", precision_at_3), ("recall_at_3", recall_at_3)]


