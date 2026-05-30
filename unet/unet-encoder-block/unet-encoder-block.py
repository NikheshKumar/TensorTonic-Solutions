import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here
    x = np.asarray(x, dtype=np.float64)

    B,H,W,C = x.shape

    C_out = out_channels

    skip_H, skip_W = H-4, W-4

    pool_out = np.zeros((B, (H-4)//2, (W-4)//2, C_out), dtype=np.float64)

    skip_out = np.zeros((B, skip_H, skip_W, C_out), dtype=np.float64)


    return pool_out, skip_out

    
