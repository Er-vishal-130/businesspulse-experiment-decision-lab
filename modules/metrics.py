from __future__ import annotations

import pandas as pd


def calculate_conversion_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def calculate_absolute_lift(control_rate: float, treatment_rate: float) -> float:
    return float(treatment_rate - control_rate)


def calculate_relative_lift(control_rate: float, treatment_rate: float) -> float:
    if abs(control_rate) < 1e-12:
        return 0.0
    return float(((treatment_rate - control_rate) / control_rate) * 100.0)


def summarize_group_metrics(df: pd.DataFrame) -> dict:
    if df.empty or "group" not in df.columns or "converted" not in df.columns:
        return {
            "control_users": 0,
            "treatment_users": 0,
            "control_conversion": 0.0,
            "treatment_conversion": 0.0,
            "absolute_lift": 0.0,
            "relative_lift": 0.0,
            "total_users": 0,
            "total_sessions": None,
            "total_records": 0,
            "overall_conversion": 0.0,
            "total_conversions": 0,
            "revenue": None,
        }

    control = df[df["group"] == "Control"].copy()
    treatment = df[df["group"] == "Treatment"].copy()

    conversion = pd.to_numeric(df["converted"], errors="coerce")
    valid = conversion.isin([0, 1])
    control_conversions = pd.to_numeric(control["converted"], errors="coerce")
    treatment_conversions = pd.to_numeric(treatment["converted"], errors="coerce")
    control_valid = control_conversions[control_conversions.isin([0, 1])]
    treatment_valid = treatment_conversions[treatment_conversions.isin([0, 1])]
    control_cr = calculate_conversion_rate(control_valid) if not control_valid.empty else 0.0
    treatment_cr = calculate_conversion_rate(treatment_valid) if not treatment_valid.empty else 0.0
    absolute_lift = calculate_absolute_lift(control_cr, treatment_cr)
    relative_lift = calculate_relative_lift(control_cr, treatment_cr)
    overall = calculate_conversion_rate(conversion[valid]) if valid.any() else 0.0
    total_users = int(df["user_id"].nunique()) if "user_id" in df.columns else int(len(df))
    control_users = int(control["user_id"].nunique()) if "user_id" in control.columns else int(len(control))
    treatment_users = (
        int(treatment["user_id"].nunique()) if "user_id" in treatment.columns else int(len(treatment))
    )
    total_sessions = int(df["session_id"].nunique()) if "session_id" in df.columns else None
    revenue = (
        float(pd.to_numeric(df["revenue"], errors="coerce").sum(min_count=1))
        if "revenue" in df.columns
        else None
    )

    return {
        "control_users": control_users,
        "treatment_users": treatment_users,
        "control_conversion": control_cr,
        "treatment_conversion": treatment_cr,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_records": int(len(df)),
        "overall_conversion": overall,
        "total_conversions": int(conversion[valid].sum()),
        "revenue": revenue,
    }


def estimate_business_impact(summary: dict, monthly_visitors: float, avg_revenue_per_conversion: float) -> dict:
    lift_points = max(summary.get("absolute_lift", 0.0), 0.0)
    additional_conversions = monthly_visitors * lift_points
    additional_revenue = additional_conversions * avg_revenue_per_conversion
    return {
        "additional_conversions": float(additional_conversions),
        "additional_revenue": float(additional_revenue),
        "monthly_visitors": float(monthly_visitors),
        "avg_revenue_per_conversion": float(avg_revenue_per_conversion),
    }


def calculate_experiment_health_score(summary: dict, p_value: float, alpha: float = 0.05, guardrail_risk: dict | None = None) -> int:
    total_users = max(summary.get("total_users", 0), 1)
    score = 50
    if p_value < alpha:
        score += 20
    else:
        score -= 15

    lift = abs(summary.get("absolute_lift", 0.0))
    score += min(int(lift * 1000), 20)

    if total_users >= 500:
        score += 10
    elif total_users < 200:
        score -= 15

    if guardrail_risk and guardrail_risk.get("critical"):
        score -= 25

    return max(0, min(100, score))


def build_business_scenario_table(base_visitors: float, lift_rate: float, revenue_per_conversion: float, multipliers: list[float]) -> pd.DataFrame:
    rows = []
    for multiplier in multipliers:
        visitors = base_visitors * multiplier
        additional_conversions = visitors * max(lift_rate, 0.0)
        revenue = additional_conversions * revenue_per_conversion
        rows.append(
            {
                "Scenario": f"{multiplier:.1f}x baseline",
                "Visitors": float(visitors),
                "Additional conversions": float(additional_conversions),
                "Additional revenue": float(revenue),
            }
        )
    return pd.DataFrame(rows)
