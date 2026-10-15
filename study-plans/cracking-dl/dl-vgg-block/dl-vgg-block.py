import numpy as np

def vgg_block(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns both ReLU convolution outputs and the pooled output.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)


    def conv_block(x, weight, bias):

        C_in, H, W = x.shape
        C_out, _, k_H, k_W = weight.shape
        stride = 1
        H_out = (H - k_H) // stride + 1
        W_out = (W - k_W) // stride + 1
    
        out = np.zeros((C_out, H_out, W_out), dtype=np.float64)
    
        for i in range(k_H):
            for j in range(k_W):
                patch = x[:, i:i + H_out * stride:stride, j:j + W_out * stride:stride]
                out += np.tensordot(weight[:, :, i, j], patch, axes=([1], [0]))
    
        out += bias[:, None, None]

        return out   

    # conv 1
    
    A1 = conv_block(x, W1, b1)
    A1_relu = np.maximum(0.0, A1)

    # conv 2

    A2 = conv_block(A1_relu, W2, b2)
    A2_relu = np.maximum(0.0, A2)

    # pooling

    s = 2
    p = 0
    k = 2
    A3 = np.pad(A2_relu, ((0,0), (p,p), (p,p)), constant_values = -float("inf"))

    C_in, Hp, Wp = A3.shape
    H_out, W_out = (Hp - k)//s + 1, (Wp - k)//s + 1

    windows = np.lib.stride_tricks.sliding_window_view(A3, (k,k), axis=(-2,-1))
    windows = windows[:, ::s, ::s, :, :]

    pool = windows.max(axis=(-2,-1))


    return {"conv1":np.round(A1_relu, 4).tolist(), "conv2":np.round(A2_relu, 4).tolist(), "pool":np.round(pool, 4).tolist()}

    
         
