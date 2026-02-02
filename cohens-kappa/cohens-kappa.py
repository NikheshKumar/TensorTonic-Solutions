def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    import numpy as np 

    rater1, rater2 = np.asarray(rater1), np.asarray(rater2)
    n = len(rater1)
    po = np.mean(rater1 == rater2)

    pe = 0.0

    labels = np.unique(np.concatenate([rater1, rater2]))

    for label in labels:
        pe += np.mean(rater1==label) * np.mean(rater2==label)

    num = po - pe
    den = 1 - pe

    if pe == 1.0:
        return 1.0

    return num/den