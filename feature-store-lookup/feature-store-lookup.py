def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here

    ans = []

    for req in requests:
        user = req["user_id"]
        offline = feature_store.get(user, defaults)
        online = req.get("online_features", {})
        merged = {**offline, **online}

        ans.append(merged)

    return ans       