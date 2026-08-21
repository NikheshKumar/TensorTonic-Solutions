import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    df =pd.DataFrame(data)
    
    index_values = df[index_col]
    columns = (df.set_index(index_col)).columns.tolist()

    output = df[columns].to_dict("list")

    return {"index_values":index_values, "columns":columns, "data": output}