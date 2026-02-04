def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here

    import numpy as np 

    ans = np.zeros(256)
    image = np.asarray(image)

    for i in range(256):
        ans[i] = np.sum(image==i)

    return ans.tolist()

