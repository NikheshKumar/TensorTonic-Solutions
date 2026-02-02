def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    import numpy as np 

    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    n, d_in = X.shape
    d_in, d_out = W.shape

    Y = [[0.0 for _ in range(d_out)]for _ in range(n)]
    for i in range(n):
        for j in range(d_out):
            Y[i][j] = sum( X[i][k] * W[k][j] for k in range(d_in) ) + b[j]

    return Y