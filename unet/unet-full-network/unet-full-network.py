import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    x = np.asarray(x, dtype=np.float64)

    B, H, W, C_in = x.shape

    h_e, w_e = (H-4)//2, (W-4)//2

    for i in range(3):
        h_e = (h_e-4)//2
        w_e = (w_e-4)//2


    h_d, w_d = h_e - 4, w_e - 4
    
    for i in range(4):
        h_d = 2*h_d - 4
        w_d = 2*w_d - 4
        

    out_shape = B, h_d, w_d, num_classes

    output = np.zeros((out_shape), dtype=np.float64)

    return output
