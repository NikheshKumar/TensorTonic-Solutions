import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    
    indices = np.arange(len(X), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)


    for i in range(0, len(X), batch_size):
        if drop_last==True and len(indices) - i < batch_size:
            break

        X_batch = X[indices[i:i + batch_size]]
        y_batch = y[indices[i:i + batch_size]]
        
        yield X_batch, y_batch