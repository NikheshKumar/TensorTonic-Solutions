import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """

    def d(a,b,c):
        return a*b + c

    grad_da = ( d(a+h, b,c) - d(a,b,c)) / h
    grad_db = ( d(a,b+h, c) - d(a,b,c)) / h
    grad_dc = ( d(a,b,c+h) - d(a,b,c)) / h

    return d(a,b,c), grad_da, grad_db, grad_dc
