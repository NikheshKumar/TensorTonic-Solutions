import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    df = pd.DataFrame(data)
    
    columns_before_reset = df.set_index(index_col).columns.tolist()
    columns_after_reset = (df.set_index(index_col)).reset_index(index_col).columns.tolist()

    return [columns_before_reset, columns_after_reset]