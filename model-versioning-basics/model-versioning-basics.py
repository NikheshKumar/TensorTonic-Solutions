def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    # Write code here
    ans = sorted(models, key=lambda x:(x['accuracy'], -x['latency'],x['timestamp']), reverse=True)

    return ans[0]['name']