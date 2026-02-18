def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    import numpy as np 

    rec_unique = set()

    for i in recommendations:
      rec_unique.update(i) 

    coverage = len(rec_unique) / n_items

    return float(coverage)

  