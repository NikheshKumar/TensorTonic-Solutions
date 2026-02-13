def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    import numpy as np 

    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)

    ref = reference_counts / np.sum(reference_counts)
    prod = production_counts / np.sum(production_counts)

    tvd = np.sum( abs(ref-prod) ) / 2

    ans = {"score":tvd, "drift_detected":bool(tvd>threshold)}

    return ans
