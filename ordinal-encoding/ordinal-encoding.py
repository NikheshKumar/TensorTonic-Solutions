def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    import numpy as np 
    values = np.asarray(values)
  
    enc = {cla:i for i,cla in enumerate(ordering)}

    res = [enc.get(v) for v in values]

    return res