import numpy as np

def hessian_convexity_analysis(X: list, y: list) -> dict:
    """
    Returns the half-MSE Hessian, eigenvalues, and convexity.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    res = {}

    n, d = X.shape

    H = X.T @ X / n

    res["hessian"] = H.tolist()

    evals = np.linalg.eigvalsh(H)

    evals = np.sort(evals)

    res["eigenvalues"] = evals.tolist()

    eps = -1e-10

    if np.any(evals >= eps):
        res["is_convex"] = bool(1==1)
    else:
        res["is_convex"] = bool(1==0)

    return res

    