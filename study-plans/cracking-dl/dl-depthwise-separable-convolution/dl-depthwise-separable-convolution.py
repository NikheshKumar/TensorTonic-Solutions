import numpy as np

def depthwise_separable_conv(x: list, W_dw: list, b_dw: list, W_pw: list, b_pw: list) -> dict:
    """
    Returns the depthwise and pointwise activations.
    """
    x = np.array(x, dtype=np.float64)
    W_dw = np.array(W_dw, dtype=np.float64)
    b_dw = np.array(b_dw, dtype=np.float64)
    W_pw = np.array(W_pw, dtype=np.float64)
    b_pw = np.array(b_pw, dtype=np.float64)

    C_in, H, W = x.shape
    _, kh, kw= W_dw.shape

    H_out = H-2
    W_out = W-2

    dw = np.zeros((C_in, H_out, W_out), dtype=np.float64)
    
    # for c in range(C_in):
    #     for i in range(H_out):
    #         for j in range(W_out):
    #             patch = x[c, i:i+kh, j:j+kw]
    #             dw[c,i,j] = np.sum(patch * W_dw[c]) + b_dw[c]

    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(-2, -1))
    dw = np.einsum("chwuv,cuv->chw", windows, W_dw)
    dw += b_dw[:, None, None]

    dw = np.maximum(0.0, dw)

    C_out, C_in = W_pw.shape
    H_out = H-2
    W_out = W-2
    
    pw = np.zeros((C_out, H_out, W_out), dtype=np.float64)

    pw = np.einsum("oc, cij->oij", W_pw, dw) + b_pw[:, None, None]
    pw = np.maximum(0.0, pw)

    return {"depthwise":np.round(dw,4).tolist(), "pointwise":np.round(pw,4).tolist()}
