import numpy as np


def transposed_conv2d(input, filters, bias=None, stride=1, padding=0):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) with values rounded to 4 decimal places.
    """
    input = np.asarray(input, dtype=np.float64)
    filters = np.asarray(filters, dtype=np.float64)
    
    C_in, H_in, W_in = input.shape
    C_in_check, C_out, k_H, k_W = filters.shape


    H_scatter = (H_in - 1) * stride + 1
    W_scatter = (W_in - 1) * stride + 1

    scatter = np.zeros((C_in, H_scatter, W_scatter), dtype=np.float64)
    scatter[:, ::stride, ::stride] = input

    H_full = H_scatter + k_H - 1
    W_full = W_scatter + k_W - 1

    output = np.zeros((C_out, H_full, W_full), dtype=np.float64)

    for kh in range(k_H):
        for kw in range(k_W):
            output[:, kh:kh + H_scatter, kw:kw + W_scatter] += np.einsum(
                "ihw,in->nhw",
                scatter,
                filters[:, :, kh, kw]
            )
            
            
    if padding > 0:
        output = output[:, padding:-padding, padding:-padding]
        
    if bias is not None:
        output += np.asarray(bias, dtype=np.float64).reshape(-1, 1, 1)


    return output

  