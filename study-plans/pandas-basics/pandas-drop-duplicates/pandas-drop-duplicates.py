import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)

    rows_before = len(df)

    cleaned_data = df.drop_duplicates()

    rows_after = len(cleaned_data)


    return [rows_before, rows_after, cleaned_data.to_dict(orient="list")]
    