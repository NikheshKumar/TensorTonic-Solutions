def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    import numpy as np

    values = np.asarray(values)
    indices = np.arange(degree+1)

    col = values.reshape(-1,1)

    phi = col ** indices

    return phi.tolist()