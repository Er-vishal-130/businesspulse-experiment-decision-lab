from __future__ import annotations


def calculate_effect_size(control_df, treatment_df) -> dict:
    control_rate = float(control_df["converted"].mean()) if not control_df.empty else 0.0
    treatment_rate = float(treatment_df["converted"].mean()) if not treatment_df.empty else 0.0
    difference = treatment_rate - control_rate
    relative_risk = (treatment_rate / control_rate) if control_rate > 0 else None
    control_odds = control_rate / (1 - control_rate) if 0 < control_rate < 1 else None
    treatment_odds = treatment_rate / (1 - treatment_rate) if 0 < treatment_rate < 1 else None
    odds_ratio = None if control_odds is None or treatment_odds is None else treatment_odds / control_odds

    return {
        "difference_in_proportions": float(difference),
        "relative_risk": relative_risk,
        "odds_ratio": odds_ratio,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
    }
