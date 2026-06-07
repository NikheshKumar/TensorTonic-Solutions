import numpy as np

def conv2d(input, filters, bias=None, padding=0, stride=1):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) with values rounded to 4 decimal places.
    """
    input = np.asarray(input, dtype=np.float64)
    filters = np.asarray(filters, dtype=np.float64)
    
    C_in, H, W = input.shape
    C_out, _, k_H, k_W = filters.shape

    if padding > 0:
        input = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        C_in, H, W = input.shape

    H_out = (H - k_H) // stride + 1
    W_out = (W - k_W) // stride + 1

    windows = np.lib.stride_tricks.sliding_window_view(input, window_shape=(k_H, k_W), axis=(1, 2))

    windows = windows[:, ::stride, ::stride, :, :]
    
    output = np.einsum('ijklm,nilm->njk', windows, filters)


    if bias:
        bias = np.array(bias, dtype=np.float64)
        output += bias.reshape(-1, 1, 1)
        
    return output
    