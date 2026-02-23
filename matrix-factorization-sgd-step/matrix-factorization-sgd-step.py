def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Write code here
    import numpy as np

    U = list(U)
    V = list(V)
    k = len(U)
    
    e = r - sum(U[i] * V[i] for i in range(k))

    U_new, V_new = U.copy(), V.copy()

    U_new = [ U[i] + lr * (e * V[i] - reg * U[i]) for i in range(k) ]
    V_new = [ V[i] + lr * (e * U[i] - reg * V[i]) for i in range(k) ]

    return U_new, V_new
  