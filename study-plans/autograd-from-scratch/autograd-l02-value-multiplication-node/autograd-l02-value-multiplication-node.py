import numpy as np

def value_multiplication_node(left, right, output_id):
    """
    Returns: a multiplication node that retains the two supplied leaf records as ordered parents
    """
    output_data = left['data'] * right['data']

    output = {"id":output_id, 'data':output_data, 'grad':0.0,"op":"*", "parents":[left, right]}

    return output
