import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)

    ans = {'rows':df.shape[0], 'cols':df.shape[1], 'columns':df.columns.to_list(), 'dtypes':{cols:str(dtype) for cols, dtype in df.dtypes.items()}, 'total_values':int(df.size)}

    return ans