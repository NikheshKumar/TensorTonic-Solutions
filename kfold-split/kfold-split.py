import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    # Write code here
    indices = np.arange(0,N,dtype=np.int64)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    
    folds = np.array_split(indices, k)

    res = []

    for i in range(k):
        curr_val_idx = folds[i]
        
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        res.append({"train_idx":train_idx, "val_idx":curr_val_idx})

    return res
        