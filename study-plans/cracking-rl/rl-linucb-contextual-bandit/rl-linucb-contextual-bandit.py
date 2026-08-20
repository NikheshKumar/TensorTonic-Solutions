def linucb_ucb(A_list, b_list, context, alpha):
    """
    Returns: list of K LinUCB scores, each rounded to 4 decimals
    """
    import numpy as np 

    A_list = np.asarray(A_list)
    b_list = np.asarray(b_list)
    context = np.asarray(context)

    linUCB = []


    for a in range(len(A_list)):

        inv_A_a = np.linalg.inv(A_list[a])
        theta_a = inv_A_a @ b_list[a]

        ucb_a = theta_a @ context + alpha * np.sqrt(context @ inv_A_a @ context)

        linUCB.append(np.round(ucb_a,4))


    return linUCB

    
