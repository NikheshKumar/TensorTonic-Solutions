import numpy as np

def kfold_cv(X: list, y: list, model_fn, k: int = 5, seed: int = 42) -> tuple:
    """
    Returns fold accuracies and their mean.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    rng = np.random.RandomState(seed)

    indices = np.arange(len(X))

    rng.shuffle(indices)

    folds = np.array_split(indices, k)

    ans = []

    for i in range(k):

        val_fold = folds[i]
        concat_folds = np.concatenate([folds[j] for j in range(k) if j!=i])
        
        predict = model_fn(X[concat_folds], y[concat_folds])
        preds = np.array(predict(X[val_fold]))
        
        val_labels = y[val_fold]
        acc = np.mean(preds == val_labels)
        ans.append(round(acc,4))


    return [ans, np.mean(ans)]
