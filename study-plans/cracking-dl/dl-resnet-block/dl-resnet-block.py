import numpy as np

def resnet_block(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns the two convolution results and residual output.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)

    def conv_block(x, weight, bias):

        C_in, H, W = x.shape
        C_out, _, k_H, k_W = W1.shape
        stride = 1
        padding = 1
        H_out = (H + 2 * padding - k_H) // stride + 1
        W_out = (W + 2 * padding - k_W) // stride + 1
    
        out = np.zeros((C_out, H_out, W_out), dtype=np.float64)
        x_new = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
        for i in range(k_H):
            for j in range(k_W):
                patch = x_new[:, i:i + H_out * stride:stride, j:j + W_out * stride:stride]
                out += np.tensordot(weight[:, :, i, j], patch, axes=([1], [0]))
    
        out += bias[:, None, None]
    
        return out


    # Conv1 block

    A1 = conv_block(x, W1, b1)

    A1_relu = np.maximum(0.0, A1)

    # Conv2 block

    Z2 = conv_block(A1_relu, W2, b2)

    Y = np.maximum(0.0, Z2+x)

    return {"conv1":np.round(A1_relu,4).tolist(), "conv2":np.round(Z2,4).tolist(), "output":Y.tolist()}
    

    

    
