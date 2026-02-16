def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image)
    H, W = image.shape
    angle = np.radians(angle_degrees)

    cx = (W-1) / 2.0
    cy = (H-1) / 2.0

    src = np.zeros_like(image)

    for i in range(H):
      for j in range(W):
        src_x = int(round(cx - (i-cy)*np.sin(angle) + (j-cx)*np.cos(angle) ) )
        src_y = int( round(cy + (i-cy)*np.cos(angle) + (j-cx)*np.sin(angle) ) )
        src[i,j] = image[src_y,src_x]

    return src.tolist()    
        