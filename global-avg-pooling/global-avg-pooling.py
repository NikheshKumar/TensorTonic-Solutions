import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x, np.float64)

    if x.ndim not in [3,4]:
        raise ValueError(f"Expected input with 3 or 4 dimensions, instead received {x.ndim}")

    H,W = x.shape[-2], x.shape[-1]

    gap = np.sum(x, axis=(-2,-1)) / (H*W)

    return gap.astype(np.float64)