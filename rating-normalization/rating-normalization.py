def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    # Write code here
    mat_normalized = []

    for row in matrix:
        r = [ele for ele in row if ele!=0]
        mean = sum(r)/len(r) if len(r)!=0 else 0.0
        mat_normalized.append([ele - mean if ele != 0 else 0.0 for ele in row])

    return mat_normalized