def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, int)

    his = np.zeros((256,), np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = image[i,j]
            his[val] += 1


    return his.tolist()