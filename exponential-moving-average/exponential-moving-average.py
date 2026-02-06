def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values, float)

    EMA = []
    EMA.append(values[0])

    for i in range(1,len(values)):
        ema = alpha * values[i] + (1 - alpha) * EMA[-1]
        EMA.append(ema)

    return EMA    


    

    