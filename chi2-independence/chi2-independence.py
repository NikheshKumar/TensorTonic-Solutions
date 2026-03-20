import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.asarray(C)

    C_rows = np.sum(C, axis=1) 
    C_cols = np.sum(C, axis=0)

    N = np.sum(C)
    
    E = np.outer(C_rows, C_cols) / N

    chi2 = np.sum((C-E)**2/E)


    return chi2, E