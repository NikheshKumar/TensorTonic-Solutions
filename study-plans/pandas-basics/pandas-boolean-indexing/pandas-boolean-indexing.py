import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)

    filtered_data = df[df[column]>threshold]
    count = len(filtered_data)

    return {"filtered_data":filtered_data.to_dict("list"), "count":count}