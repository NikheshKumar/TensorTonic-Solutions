def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, dtype=float)
    out = np.zeros((new_h, new_w), dtype=float)

    H, W = image.shape

    for i in range(new_h):
        for j in range(new_w):

            src_x = i * (H-1)/(new_h-1) if new_h > 1.0 else 0.0
            src_y = j * (W-1)/(new_w-1) if new_w > 1.0 else 0.0

            i0 = int(np.floor(src_x))
            i1 = min(i0 + 1, H - 1)
            j0 = int(np.floor(src_y))
            j1 = min(j0 + 1, W - 1)
            
            di = src_x-i0
            dj = src_y-j0 

            out[i, j] = (
                image[i0, j0] * (1 - di) * (1 - dj) +
                image[i1, j0] * di * (1 - dj) +
                image[i0, j1] * (1 - di) * dj +
                image[i1, j1] * di * dj
            )


    return out.tolist()

            
            

    