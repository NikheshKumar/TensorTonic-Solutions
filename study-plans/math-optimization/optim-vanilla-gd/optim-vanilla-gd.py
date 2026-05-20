import numpy as np

def vanilla_gradient_descent(x0, y0, lr, n_iters):
    """
    Returns: dict with 'trajectory' (list of [x,y] pairs), 'final_point' ([x,y]), 'final_value' (float)
    """
    x = x0 
    y = y0 

    traj = [[x0,y0]]

    for i in range(n_iters):
        x = (1-2*lr)*x
        y = (1-6*lr)*y

        traj.append([x,y])
        
    final_point = [x,y]
    final_value = (x**2) + 3*(y**2)
    
    return {'trajectory':traj, 'final_point':final_point, 'final_value':final_value} 