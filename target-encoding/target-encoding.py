def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    import numpy as np 

    categories = np.asarray(categories)
    targets = np.asarray(targets)

    n = len(categories)

    if n==0:
        return []

    t_sums = {}
    count = {}

    for i, cat in enumerate(categories):
        if cat not in t_sums:
            t_sums[cat] = 0.0
            count[cat] = 0
            
        t_sums[cat] += float(targets[i])
        count[cat] += 1

    ans = [float(t_sums[cat]/count[cat]) for cat in categories]

    return ans