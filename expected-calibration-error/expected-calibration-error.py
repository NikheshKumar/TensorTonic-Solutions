def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    import numpy as np 

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    N = len(y_true)

    cla, counts = np.unique(y_pred, return_counts=True)

    B = np.linspace(0, 1, n_bins + 1)

    indices = np.digitize(y_pred, B) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    bin_counts = np.bincount(indices, minlength=n_bins)

    bin_acc_sum = np.bincount(indices, weights=y_true, minlength=n_bins)
    bin_conf_sum = np.bincount(indices, weights=y_pred, minlength=n_bins)

    mask = bin_counts > 0.0

    acc = bin_acc_sum[mask] / bin_counts[mask]
    conf = bin_conf_sum[mask] / bin_counts[mask]


    ece = np.sum( bin_counts[mask] / N * np.abs(acc - conf))

    return ece