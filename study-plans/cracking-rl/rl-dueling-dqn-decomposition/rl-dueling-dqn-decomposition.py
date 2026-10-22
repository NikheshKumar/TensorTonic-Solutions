def dueling_q_values(V: list, A_advantages: list) -> list:
    """
    Returns the Q-value matrix as a nested list, rounded to four decimals.
    """
    Q = []
    

    for v,row_a in zip(V, A_advantages):
        M = len(row_a)
        row_mean = sum(row_a) / M
        q = [v+a-row_mean for a in row_a ]
        Q.append(q)

    return [[round(float(q),4) for q in row] for row in Q]
            


            