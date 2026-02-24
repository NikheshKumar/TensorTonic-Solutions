def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    import numpy as np 
    from collections import defaultdict

    categories = np.asarray(categories)
    targets = np.asarray(targets)

    d = defaultdict(float)
    c = defaultdict(float)

    for item, t in zip(categories, targets):
      d[item] += t
      c[item] += 1

    ans = [ d[item]/c[item] for item in categories]

    return ans