def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    # Write code here
    import numpy as np

    image = np.asarray(image, dtype=float)
    H, W = image.shape

    image_padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    out = np.zeros((H, W), dtype=float)

    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    ky = kx.T

    for i in range(H):
        for j in range(W):

            window = image_padded[i:i+3, j:j+3]

            Gx = np.sum(window * kx)
            Gy = np.sum(window * ky)

            out[i][j] = (Gx * Gx + Gy * Gy) ** 0.5


    return out.tolist()
