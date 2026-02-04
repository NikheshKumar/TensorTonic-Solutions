def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, dtype=float)

    H, W, _ = image.shape

    ans = []

    for i in range(H):
        row = []
        for j in range(W):
            R, G, B = image[i][j]
            val = 0.299*R + 0.587*G + 0.114*B
            row.append(val)
        ans.append(row)    

    return ans        