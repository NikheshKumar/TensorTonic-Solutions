import numpy as np

def lenet_forward(x: list, W_conv: list, b_conv: list, W_fc1: list, b_fc1: list, W_fc2: list, b_fc2: list) -> dict:
    """
    Returns convolution, pooling, hidden, and logit outputs.
    """
    x = np.array(x, dtype=np.float64)
    W_conv = np.array(W_conv, dtype=np.float64)
    b_conv = np.array(b_conv, dtype=np.float64)
    W_fc1 = np.array(W_fc1, dtype=np.float64)
    W_fc2 = np.array(W_fc2, dtype=np.float64)
    b_fc1 = np.array(b_fc1, dtype=np.float64)
    b_fc2 = np.array(b_fc2, dtype=np.float64)

    #conv1
    added_dim = (x.ndim==3)
    if x.ndim==3:
        x = x[None, :, :,:]
        
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, kh, kw = W_conv.shape

    kh = kw = 3
    sh = sw = 1

    H_out = (H_in - kh) // sh + 1
    W_out = (W_in - kw) // sw + 1

    conv_out = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    window = x[n, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                    conv_out[n, c, i, j] = np.sum(window * W_conv[c]) + b_conv[c]

    conv_out = np.maximum(0.0, conv_out)
    
    #pooling

    sh = sw = 2
    kh = kw = 2

    N,C,H,W = conv_out.shape
    H_out = (H-kh)//sh + 1
    W_out = (W-kw)//sw + 1
    
    pool_out = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    window = conv_out[n, c, i*sh:i*sh+kh, j*sw:j*sw+kw]
                    pool_out[n, c, i, j] = np.max(window)

    #fc1

    z = pool_out.reshape(pool_out.shape[0], -1)
    fc1 = z @ W_fc1.T + b_fc1

    fc1_out = np.maximum(0.0, fc1)

    #fc2

    logits = fc1_out @ W_fc2.T + b_fc2
    
    if added_dim:
        conv_out = conv_out.squeeze(0)
        pool_out = pool_out.squeeze(0)
        fc1_out = fc1_out.squeeze(0)
        logits = logits.squeeze(0)


    return {"conv_out":conv_out, "fc1_out":fc1_out, "logits":logits, "pool_out":pool_out}
