ALERT_CONFIG = { 
    "confidence_threshold": 0.75, 
    "change_percent_threshold": 15.0, 
    "proximity_duration_threshold": 20   # frames 
} 
def check_alert(action_result, change_result, proximity_duration, config=ALERT_CONFIG): 
    triggered = ( 
        action_result["label"] == "suspicious" 
        and action_result["confidence"] >= config["confidence_threshold"] 
        and change_result["change_percent"] >= config["change_percent_threshold"] 
    )
    message = None 
    if triggered: 
        message = ( 
            f"ALERT: Suspicious activity detected (confidence {action_result['confidence']:.2f}), " 
            f"surface change {change_result['change_percent']:.1f}% — flagged for security review" 
        ) 
    return triggered, message