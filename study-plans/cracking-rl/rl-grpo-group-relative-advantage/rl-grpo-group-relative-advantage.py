def grpo_advantages(rewards, group_indices, eps=1e-8):
    """
    Returns: list of N advantages standardized within each group, rounded to 4 decimals
    """
    g_sum = {}
    count_g = {}
    var_g = {}

    for r,g in zip(rewards, group_indices):

        g_sum[g] = g_sum.get(g, 0.0) + r
        count_g[g] = count_g.get(g, 0.0) + 1
        var_g[g] = var_g.get(g, 0.0) + r ** 2
        

    gmr = {g:g_sum[g]/count_g[g] for g in g_sum}

    var = {g: (var_g[g]/count_g[g] - (gmr[g]**2)) for g in g_sum}

    sigma_g = {g:math.sqrt(max(v,0.0)) for g,v in var.items()}

    A = []

    for r, g in zip(rewards, group_indices):
        a = (r - gmr[g]) / (sigma_g[g] + eps)
        A.append(round(a, 4))


    return A

    
