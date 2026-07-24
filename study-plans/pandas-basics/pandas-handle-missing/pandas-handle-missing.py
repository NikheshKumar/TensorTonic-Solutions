import pandas as pd

def handle_missing(data, fill_value):
    """
    Returns: dict with 'null_counts' (dict) and 'cleaned_data' (dict)
    """
    df = pd.DataFrame(data)

    null_counts = df.isna().sum(axis=0).to_dict()

    cleaned_data = df.fillna(fill_value)

    return {
        'null_counts': null_counts,
        'cleaned_data': cleaned_data.to_dict(orient='list')
    }