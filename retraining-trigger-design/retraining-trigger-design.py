def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    # Write code here
    
    drift_threshold = config.get("drift_threshold")
    performance_threshold = config.get("performance_threshold")
    max_staleness = config.get("max_staleness")
    cooldown = config.get("cooldown")
    retrain_cost = config.get("retrain_cost")
    budget = config.get("budget")

    ans = []
    days_since_retrain = 0
    last_day_retrained = -cooldown

    for s in daily_stats:
        day = s["day"]
        drift_score = s["drift_score"]
        performance = s["performance"]
        days_since_retrain += 1

        drift_trigger = drift_score > drift_threshold
        performance_trigger = performance < performance_threshold
        staleness_trigger = days_since_retrain >= max_staleness

        trigger = drift_trigger or performance_trigger or staleness_trigger

        cooldown_constraint = day - last_day_retrained >= cooldown
        
        if trigger and cooldown_constraint and budget >= retrain_cost:
            budget -= retrain_cost
            days_since_retrain = 0
            last_day_retrained = day
            ans.append(day)


    return ans 