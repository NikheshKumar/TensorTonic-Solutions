import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if len(fpr)!=len(tpr) or len(fpr)<2:
      raise ValueError("Both FPR and TPR should have the same lenght and at least 2 points")

    area = np.trapezoid(tpr, x=fpr)

    return float(area)