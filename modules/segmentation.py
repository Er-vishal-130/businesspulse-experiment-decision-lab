from __future__ import annotations

from modules.confidence_intervals import calculate_difference_confidence_interval
from modules.hypothesis_testing import run_two_proportion_test
from modules.metrics import calculate_absolute_lift, calculate_conversion_rate, calculate_relative_lift


def analyze_segment(df, segment_col: str = "device") -> list[dict]:
    if segment_col not in df.columns or "group" not in df.columns or "converted" not in df.columns:
        return []

    rows = []
    values = sorted(df[segment_col].dropna().astype(str).unique())
    for value in values:
        subset = df[df[segment_col].astype(str) == value].copy()
        control = subset[subset["group"] == "Control"]
        treatment = subset[subset["group"] == "Treatment"]
        if control.empty or treatment.empty:
            continue

        control_rate = calculate_conversion_rate(control["converted"])
        treatment_rate = calculate_conversion_rate(treatment["converted"])
        abs_lift = calculate_absolute_lift(control_rate, treatment_rate)
        rel_lift = calculate_relative_lift(control_rate, treatment_rate)
        hypothesis = run_two_proportion_test(control, treatment)
        ci = calculate_difference_confidence_interval(control, treatment)

        warning = "; ".join(hypothesis["warnings"]) or "No low-count warning from the selected test."
        rows.append(
            {
                "segment_column": segment_col,
                "segment_value": str(value),
                "sample_size": int(len(subset)),
                "control_conversion": control_rate,
                "treatment_conversion": treatment_rate,
                "absolute_lift": abs_lift,
                "relative_lift": rel_lift,
                "p_value": hypothesis["p_value"],
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "method": hypothesis["method"],
                "effect_significant": bool(hypothesis["p_value"] < 0.05),
                "warning": warning,
            }
        )
    return rows
