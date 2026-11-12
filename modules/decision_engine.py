from __future__ import annotations


def decide_experiment(summary: dict, p_value: float, alpha: float = 0.05, ci_delta: dict | None = None, guardrail_risk: dict | None = None, total_users: int = 0) -> tuple[str, str]:
    if total_users < 200:
        return "INCONCLUSIVE", "Sample size is too limited to make a reliable recommendation."

    if ci_delta is not None and ci_delta.get("low", 0.0) <= 0 and p_value < alpha:
        return "NEGATIVE RESULT", "The treatment is statistically worse than control, or the confidence interval is not supporting improvement."

    if p_value < alpha and summary.get("absolute_lift", 0.0) > 0:
        if guardrail_risk and guardrail_risk.get("critical"):
            return "INCONCLUSIVE", "The effect is positive but guardrail metrics worsen, so the recommendation is not safe yet."
        if ci_delta is not None and ci_delta.get("low", 0.0) <= 0:
            return "INCONCLUSIVE", "The confidence interval includes zero, limiting confidence in a positive effect."
        return "WIN", "Recommend Treatment because there is statistically significant improvement and guardrail metrics are acceptable."

    if p_value >= alpha:
        return "NO SIGNIFICANT EVIDENCE", "Evidence is insufficient to distinguish a real effect from random variation."

    if summary.get("absolute_lift", 0.0) < 0:
        return "NEGATIVE RESULT", "The treatment performs worse than control on the primary metric."

    return "INCONCLUSIVE", "The data quality or sample size is not sufficient to support a final decision."
