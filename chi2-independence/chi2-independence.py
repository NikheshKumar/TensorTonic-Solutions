import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.asarray(C)

    row_totals = np.sum(C, axis=1)
    col_totals = np.sum(C, axis=0)

    E = np.outer(row_totals, col_totals) / np.sum(C)

    chi2 = np.sum((C-E)**2 / E)

    return chi2, E

    