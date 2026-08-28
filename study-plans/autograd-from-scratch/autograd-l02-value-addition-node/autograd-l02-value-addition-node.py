import numpy as np

def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    output_data = left['data'] + right['data']
    
    output = {"id":output_id, 'data':output_data, 'grad':0.0, 'op':'+','parents':[left, right]}

    return output
