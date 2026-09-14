import numpy as np

def vgg_features(x: np.ndarray, config: list,
                 kernels: list, biases: list) -> np.ndarray:
    """
    Returns the float64 VGG features in NHWC layout.
    """
    y = x.copy()
    idx = 0
    
    for layer in config:

        N, H, W, C = y.shape
        
        if layer == "M":
            y = y.reshape(N, H // 2, 2, W // 2, 2, C)
            y = np.max(y, axis=(2, 4))
            continue

        kernel = kernels[idx]
        bias = biases[idx]
        size = kernel.shape[0]
        pad = size // 2

        x_pad = np.pad(y, ((0,0), (pad, pad), (pad, pad), (0,0)))
        new_out = np.zeros((N, H, W, kernel.shape[3]), dtype=np.float64)
        
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                window = x_pad[:, i:i + size, j:j+size, :]
                N = window.shape[0]
                K = kernel.shape[-1]
                window_flat = window.reshape(N, -1)
                kernel_flat = kernel.reshape(-1, K)
                new_out[:, i, j, :] =  window_flat @ kernel_flat + bias

        y = np.maximum(0, new_out)
        idx += 1

    return y
            
            