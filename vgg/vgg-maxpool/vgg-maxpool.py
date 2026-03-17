import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    x = np.asarray(x, float)

    K = 2
    s = 2

    B, H_in, W_in, C_in = x.shape

    H_out = H_in // 2
    W_out = W_in // 2
    C_out = C_in

    y = np.zeros((B, H_out, W_out, C_in), float)

    for i in range(H_out):
      
      for j in range(W_out):
        
        patch = x[:,i*K:i*K+s,j*K:j*K+s,:]
        
        y[:, i, j, :] = np.max(patch, axis=(1, 2))

    return y
    