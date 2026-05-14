import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)

    ans = df[column]

    return {'values':ans.to_list(), 'length':int(ans.shape[0])}