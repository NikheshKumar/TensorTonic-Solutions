import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here

    x = np.asarray(x, np.float64)

    if x.ndim!=3 and x.ndim!=4:
        raise ValueError(f"Expected input with 3 or 4 dimensions, instead {x.ndim}")


    gap = np.mean(x, axis=(-2,-1))

    return gap.astype(np.float64)