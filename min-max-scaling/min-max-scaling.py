def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    import numpy as np 
    
    data = np.asarray(data, float)   

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    data_diff = data_max - data_min

    data_diff[data_diff==0.0] = 1.0 

    data_new = (data - data_min) / data_diff  

    return data_new.tolist()  
  
  
  