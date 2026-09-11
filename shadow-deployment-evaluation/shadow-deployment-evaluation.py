import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    # Write code here
    n = len(shadow_log)
    
    shadow_acc = sum(1 for e in shadow_log if e["prediction"] == e["actual"]) / n
    production_acc = sum(1 for e in production_log if e["prediction"] == e["actual"]) / n
    acc_gain = shadow_acc - production_acc

    latencies = sorted(e["latency_ms"] for e in shadow_log)
    i = math.ceil(0.95 * n) - 1
    i = max(0, min(i, n - 1))  
    shadow_latency = latencies[i]

    agreements = sum(
        1 for p, s in zip(production_log, shadow_log)
        if p["prediction"] == s["prediction"]
    )
    ag_rate = agreements / n

    promote = (
        acc_gain >= criteria.get("min_accuracy_gain", 0.0)
        and shadow_latency <= criteria.get("max_latency_p95", float("inf"))
        and ag_rate >= criteria.get("min_agreement_rate", 0.0)
    )

    
    return {"promote":promote, "metrics": {"shadow_accuracy": shadow_acc, "production_accuracy": production_acc, "accuracy_gain": acc_gain, "shadow_latency_p95": shadow_latency, "agreement_rate": ag_rate}}