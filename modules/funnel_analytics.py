"""
Funnel Analytics Module

Comprehensive funnel analysis with:
- Automatic funnel detection
- Drop-off analysis
- Segment-wise funnel comparison
- Conversion stage optimization
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


def analyze_conversion_funnel(
    df: pd.DataFrame,
    funnel_steps: list[str] = None,
    group_by: str = None,
) -> dict[str, Any]:
    """
    Analyze conversion funnel across specified steps.

    Args:
        df: Input dataframe
        funnel_steps: List of column names representing funnel steps (in order)
        group_by: Optional column to segment funnel by (e.g., 'group', 'device')

    Returns:
        Dictionary with funnel analysis including:
        - step_metrics: Users and conversion rate at each step
        - drop_off_analysis: Drop-off between consecutive steps
        - biggest_drop_off: Stage with largest drop
        - segment_comparison: Funnel by segment if group_by specified
    """
    if funnel_steps is None:
        # Auto-detect binary columns that could be funnel steps
        funnel_steps = _auto_detect_funnel_steps(df)

    if not funnel_steps:
        return {
            "status": "error",
            "message": "No funnel steps available. Provide binary columns (0/1) representing each stage.",
            "step_metrics": [],
            "drop_off_analysis": [],
        }

    # Validate funnel steps exist
    available_steps = [step for step in funnel_steps if step in df.columns]

    if not available_steps:
        return {
            "status": "error",
            "message": f"None of the specified steps found in dataset: {funnel_steps}",
            "step_metrics": [],
            "drop_off_analysis": [],
        }

    # Calculate funnel metrics
    if group_by and group_by in df.columns:
        funnel_by_segment = {}
        for segment_value in df[group_by].unique():
            segment_df = df[df[group_by] == segment_value]
            funnel_by_segment[str(segment_value)] = _calculate_funnel_metrics(
                segment_df, available_steps
            )

        # Also calculate overall
        overall_metrics = _calculate_funnel_metrics(df, available_steps)

        return {
            "status": "success",
            "funnel_steps": available_steps,
            "overall_metrics": overall_metrics,
            "segment_metrics": funnel_by_segment,
            "group_by": group_by,
        }
    else:
        metrics = _calculate_funnel_metrics(df, available_steps)

        return {
            "status": "success",
            "funnel_steps": available_steps,
            "overall_metrics": metrics,
            "group_by": None,
        }


def _auto_detect_funnel_steps(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect potential funnel steps from binary columns.
    """
    binary_columns = []

    for col in df.columns:
        if col in ["user_id", "session_id", "group", "date"]:
            continue

        # Check if column is binary
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        if numeric_values.isin([0, 1]).all():
            binary_columns.append(col)

    # Prioritize common funnel steps
    preferred_order = [
        "visited", "viewed", "clicked", "added_to_cart",
        "initiated_checkout", "converted", "completed"
    ]

    ordered_steps = []
    for preferred in preferred_order:
        if preferred in binary_columns:
            ordered_steps.append(preferred)
            binary_columns.remove(preferred)

    # Add remaining columns
    ordered_steps.extend(binary_columns)

    return ordered_steps


def _calculate_funnel_metrics(
    df: pd.DataFrame,
    funnel_steps: list[str],
) -> dict[str, Any]:
    """Calculate funnel metrics for a single dataframe."""
    step_metrics = []
    total_users = len(df)

    for i, step in enumerate(funnel_steps):
        step_values = pd.to_numeric(df[step], errors="coerce")
        users_at_step = int(step_values.sum())
        conversion_from_top = users_at_step / total_users if total_users > 0 else 0

        if i > 0:
            # Calculate conversion from previous step
            prev_step = funnel_steps[i - 1]
            prev_values = pd.to_numeric(df[prev_step], errors="coerce")
            users_at_prev = int(prev_values.sum())
            conversion_from_prev = users_at_step / users_at_prev if users_at_prev > 0 else 0
            drop_off_from_prev = 1 - conversion_from_prev
        else:
            conversion_from_prev = 1.0
            drop_off_from_prev = 0.0
            users_at_prev = total_users

        step_metrics.append({
            "step_number": i + 1,
            "step_name": step,
            "users": users_at_step,
            "conversion_from_top": conversion_from_top,
            "conversion_from_previous": conversion_from_prev,
            "drop_off_from_previous": drop_off_from_prev,
            "users_dropped": users_at_prev - users_at_step,
        })

    # Identify biggest drop-off
    drop_offs = [(m["step_name"], m["drop_off_from_previous"]) for m in step_metrics[1:]]
    biggest_drop_off = max(drop_offs, key=lambda x: x[1]) if drop_offs else (None, 0)

    return {
        "step_metrics": step_metrics,
        "biggest_drop_off_step": biggest_drop_off[0],
        "biggest_drop_off_rate": biggest_drop_off[1],
        "overall_conversion": step_metrics[-1]["conversion_from_top"] if step_metrics else 0,
        "total_users_entered": total_users,
        "total_users_completed": step_metrics[-1]["users"] if step_metrics else 0,
    }


def compare_funnels(
    control_funnel: dict,
    treatment_funnel: dict,
) -> dict[str, Any]:
    """
    Compare Control vs Treatment funnels.

    Returns:
        Comparison metrics showing where Treatment differs
    """
    if not control_funnel.get("step_metrics") or not treatment_funnel.get("step_metrics"):
        return {
            "status": "error",
            "message": "Both funnels must have step metrics",
        }

    control_steps = control_funnel["step_metrics"]
    treatment_steps = treatment_funnel["step_metrics"]

    comparisons = []

    for i in range(min(len(control_steps), len(treatment_steps))):
        ctrl_step = control_steps[i]
        trt_step = treatment_steps[i]

        if ctrl_step["step_name"] != trt_step["step_name"]:
            continue

        # Compare conversion rates
        ctrl_conv = ctrl_step["conversion_from_previous"]
        trt_conv = trt_step["conversion_from_previous"]
        diff = trt_conv - ctrl_conv
        pct_change = (diff / ctrl_conv * 100) if ctrl_conv > 0 else 0

        comparisons.append({
            "step_name": ctrl_step["step_name"],
            "step_number": i + 1,
            "control_conversion": ctrl_conv,
            "treatment_conversion": trt_conv,
            "absolute_difference": diff,
            "percentage_change": pct_change,
            "improved": diff > 0,
        })

    # Identify stage with biggest improvement/decline
    if comparisons:
        sorted_by_change = sorted(comparisons, key=lambda x: abs(x["absolute_difference"]), reverse=True)
        biggest_change = sorted_by_change[0]
    else:
        biggest_change = None

    return {
        "status": "success",
        "step_comparisons": comparisons,
        "biggest_change": biggest_change,
        "overall_control_conversion": control_funnel.get("overall_conversion", 0),
        "overall_treatment_conversion": treatment_funnel.get("overall_conversion", 0),
    }


def identify_funnel_bottlenecks(funnel_metrics: dict, threshold: float = 0.3) -> list[dict]:
    """
    Identify funnel bottlenecks (stages with high drop-off).

    Args:
        funnel_metrics: Funnel metrics from analyze_conversion_funnel
        threshold: Drop-off threshold to flag as bottleneck (default 30%)

    Returns:
        List of bottleneck stages with recommendations
    """
    if not funnel_metrics.get("step_metrics"):
        return []

    bottlenecks = []

    for step in funnel_metrics["step_metrics"][1:]:  # Skip first step
        if step["drop_off_from_previous"] >= threshold:
            bottlenecks.append({
                "step_name": step["step_name"],
                "drop_off_rate": step["drop_off_from_previous"],
                "users_lost": step["users_dropped"],
                "severity": "Critical" if step["drop_off_from_previous"] > 0.5 else "High",
                "recommendations": _generate_bottleneck_recommendations(
                    step["step_name"],
                    step["drop_off_from_previous"]
                ),
            })

    return bottlenecks


def _generate_bottleneck_recommendations(step_name: str, drop_off_rate: float) -> list[str]:
    """Generate recommendations for addressing bottlenecks."""
    recommendations = []

    recommendations.append(f"Investigate user experience at {step_name} stage")

    if drop_off_rate > 0.5:
        recommendations.append(f"Critical drop-off ({drop_off_rate:.1%}) - consider UX redesign")
        recommendations.append("Conduct user testing to identify friction points")
    else:
        recommendations.append(f"Significant drop-off ({drop_off_rate:.1%}) - review and optimize")

    recommendations.append(f"Analyze segment-level performance at {step_name}")
    recommendations.append(f"Test variations at {step_name} to improve conversion")

    return recommendations
