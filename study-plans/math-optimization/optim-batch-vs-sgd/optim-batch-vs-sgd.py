import numpy as np

def batch_gd_compare(X, y, batch_sizes, n_epochs, lr, seed):
    """
    Returns: list of loss curves (one list per batch size).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    N, D = X.shape
    
    ans = []

    for s in batch_sizes:
        
        rng = np.random.RandomState(seed)
        w = np.zeros((D,), dtype=np.float64)
        epoch_loss_curves = []
        
        for i in range(n_epochs):
            
            indices = rng.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for epoch_start in range(0,N,s):

                epoch_end = min(epoch_start + s, N)
                X_batch = X_shuffled[epoch_start:epoch_end]
                y_batch = y_shuffled[epoch_start:epoch_end]
                grad = 2.0 * (X_batch.T @ (X_batch @ w - y_batch)) / (epoch_end - epoch_start)
                w = w - lr * grad

            
            loss = np.mean((X @ w - y) ** 2)
            epoch_loss_curves.append(np.round(loss,6))


        ans.append(epoch_loss_curves)


    return ans
        
        
