import pandas as pd

def iloc_selection(data, row, col):
    """
    Returns: list [element, row_values, col_values]
    """
    df = pd.DataFrame(data)

    ele = df.iloc[row, col]

    row_vals = df.iloc[row,:].tolist()

    col_vals = df.iloc[:,col].tolist()

    return [ele, row_vals, col_vals]