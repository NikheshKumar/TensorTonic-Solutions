def thompson_posterior(K: int, observations: list, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> tuple:
    """
    Returns (alpha_list, beta_list), each with K floats rounded to four decimals.
    """
    alpha_list = [alpha_prior]*K
    beta_list = [beta_prior]*K

    for arm_ind, reward in observations:
        alpha_list[arm_ind] += 1 if reward==1 else 0
        beta_list[arm_ind] += 1 if reward==0 else 0

    return [round(float(a),4) for a in alpha_list], [round(float(b),4) for b in beta_list]

        
        
        