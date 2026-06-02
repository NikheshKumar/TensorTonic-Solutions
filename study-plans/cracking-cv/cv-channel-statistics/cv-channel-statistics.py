def channel_statistics(batch):
    """
    Returns: dict with keys "mean" and "std", each a list of length C, with every entry rounded to 4 decimals.
    """
    import numpy as np 

    batch = np.asarray(batch, dtype=np.float64)

    m = np.mean(batch, axis=(0,1,2))

    std = np.std(batch, axis=(0,1,2))

    return {"mean":m, "std":std}
