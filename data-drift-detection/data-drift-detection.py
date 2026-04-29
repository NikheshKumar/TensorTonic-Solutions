def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    import numpy as np 

    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)

    r = reference_counts / np.sum(reference_counts)
    p = production_counts / np.sum(production_counts)

    tvd = np.sum(np.abs(r-p)) / 2

    ans_drift = {"score":tvd, "drift_detected":bool(tvd > threshold)}

    return ans_drift