def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here

    import numpy as np 

    matrix = np.asarray(matrix)

    mask = np.where(matrix>0, matrix, np.nan)
    row_means = np.nanmean(mask, axis=1, keepdims=True)
    mat_norm = np.where(matrix>0, matrix-row_means, 0.0)

    return mat_norm.tolist()
  