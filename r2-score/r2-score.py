import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here


    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred,float)

    m = np.mean(y_true)

    SS_res = np.sum(( y_pred-y_true)**2 )
    SS_tot = np.sum( (y_true-m)**2 )

    if SS_tot == 0.0:
      if np.allclose(y_pred, y_true):
            return 1.0
      else:
          return 0.0

    return (1.0 - SS_res / SS_tot)        