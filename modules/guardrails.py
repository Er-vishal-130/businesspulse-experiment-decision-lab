from __future__ import annotations

import pandas as pd


def compute_guardrail_summary(df: pd.DataFrame) -> dict:
    if df.empty or "group" not in df.columns:
        return {"Control": {}, "Treatment": {}}

    summary = {}
    for group_name in ["Control", "Treatment"]:
        subset = df[df["group"] == group_name]
        if subset.empty:
            summary[group_name] = {}
            continue
        metrics = {
            "revenue_per_user": "revenue",
            "bounce_rate": "bounce",
            "error_rate": "error_flag",
            "avg_session_duration": "session_duration",
            "pages_per_session": "pages_viewed",
        }
        summary[group_name] = {
            label: float(pd.to_numeric(subset[column], errors="coerce").mean())
            for label, column in metrics.items()
            if column in subset.columns and pd.to_numeric(subset[column], errors="coerce").notna().any()
        }
    return summary


def evaluate_guardrail_risk(summary: dict) -> dict:
    control = summary.get("Control", {})
    treatment = summary.get("Treatment", {})
    flags = []
    if (
        "error_rate" in treatment
        and "error_rate" in control
        and treatment["error_rate"] > control["error_rate"] * 1.10
    ):
        flags.append("Error rate increased materially.")
    if (
        "bounce_rate" in treatment
        and "bounce_rate" in control
        and treatment["bounce_rate"] > control["bounce_rate"] * 1.10
    ):
        flags.append("Bounce rate increased materially.")
    if (
        "revenue_per_user" in treatment
        and "revenue_per_user" in control
        and treatment["revenue_per_user"] < control["revenue_per_user"] * 0.95
    ):
        flags.append("Revenue per user declined compared with Control.")
    return {
        "flags": flags,
        "critical": bool(flags),
    }
