def rloo_advantages(rewards, group_indices):
    """
    Returns: list of N RLOO advantages rounded to 4 decimals
    """
    tot = sum(rewards)

    A = []

    group_sum = {}
    group_count = {}

    for r,g in zip(rewards, group_indices):
        group_sum[g] = group_sum.get(g, 0.0) + r
        group_count[g] = group_count.get(g, 0.0) + 1

    for r,g in zip(rewards, group_indices):
        a = r - (group_sum[g]-r)/(group_count[g]-1)
        A.append(round(a,4))

    return A
        
