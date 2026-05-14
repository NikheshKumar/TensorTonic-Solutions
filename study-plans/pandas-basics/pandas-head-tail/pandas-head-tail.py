import pandas as pd

def head_tail(data, n):
    """
    Returns: dict with 'head' and 'tail' (both dicts of column -> list)
    """
    df = pd.DataFrame(data)

    head = df.head(n)
    tail = df.tail(n)

    return {'head':head.to_dict("list"), 'tail':tail.to_dict("list")}