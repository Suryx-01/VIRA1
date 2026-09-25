def calculate_risk(
    deepfake_probability,
    speaker_match_probability,
    context_risk
):
    """
    Calculate the overall VIRA risk score.

    All inputs should be between 0 and 1.
    """

    # Higher deepfake probability = higher risk
    deepfake_risk = deepfake_probability

    # Lower speaker match = higher risk
    speaker_risk = 1 - speaker_match_probability

    # Context risk already represents risk
    context_risk_value = context_risk

    # VIRA risk fusion
    final_risk = (
        0.45 * deepfake_risk
        + 0.35 * speaker_risk
        + 0.20 * context_risk_value
    )

    # Keep score between 0 and 1
    final_risk = max(0, min(final_risk, 1))

    # Determine action
    if final_risk >= 0.80:
        action = "Escalate"
    elif final_risk >= 0.50:
        action = "Hold"
    elif final_risk >= 0.30:
        action = "Monitor"
    else:
        action = "Allow"

    return {
        "final_risk_score": final_risk,
        "recommended_action": action
    }