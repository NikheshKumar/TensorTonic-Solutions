def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, float)

    flat_image = image.flatten()

    his, bins = np.histogram(flat_image, bins=256, range=(0, 255))

    cdf = his.cumsum()

    cdf_masked = np.ma.masked_equal(cdf, 0)

    cdf_min = np.min(cdf_masked)
    tot = np.max(cdf_masked)

    cdf_norm = np.round( (cdf_masked - cdf_min) * 255 / (tot - cdf_min))

    ans = np.ma.filled(cdf_norm, 0).astype('uint8')

    ans = ans[flat_image.astype(int)].reshape(image.shape)


    return ans.tolist()
    