import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """

    df = pd.DataFrame(data)

    ans = {'dtypes':df.dtypes.astype(str).to_dict(), 'type_counts':df.dtypes.astype(str).value_counts().to_dict(), "num_columns":len(df.columns)}

    return ans