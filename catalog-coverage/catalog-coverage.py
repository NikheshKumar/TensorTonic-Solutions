def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    if n_items==0:
        return 0.0
    else:
        unique = set()
        for i in recommendations:
          unique.update(i) 
        return len(unique) / n_items