def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    import numpy as np 

    ans = sorted(models, key=lambda x: (x['accuracy'], -x['latency'], x['timestamp']), reverse=True)

    return ans[0]['name']