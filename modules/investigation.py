"""
Enhanced Root Cause Investigation Module

Provides advanced root cause analysis workflow:
- Multi-dimensional drill-down
- Contribution ranking with transparent methodology
- Investigation timeline tracking
- Evidence-based next steps
- Sub-segment analysis
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


def enhanced_root_cause_investigation(
    df: pd.DataFrame,
    metric_col: str = "converted",
    dimensions: list[str] = None,
    min_observations: int = 30,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Conduct comprehensive root cause investigation across multiple dimensions.

    Args:
        df: Input dataframe
        metric_col: Metric column to investigate
        dimensions: List of dimensions to analyze
        min_observations: Minimum observations per segment
        top_n: Number of top contributors to return

    Returns:
        Dictionary with investigation results including:
        - dimension_rankings: Top contributing dimensions
        - top_segments: Top contributing segments across all dimensions
        - investigation_timeline: Suggested investigation order
        - evidence_summary: Aggregated evidence
        - next_steps: Recommended actions
    """
    if dimensions is None:
        dimensions = ["device", "traffic_source", "country", "gender", "page_type"]

    # Filter to available dimensions
    available_dimensions = [dim for dim in dimensions if dim in df.columns]

    if not available_dimensions:
        return {
            "status": "error",
            "message": "No valid dimensions available for investigation",
            "dimension_rankings": [],
            "top_segments": [],
            "investigation_timeline": [],
            "evidence_summary": {},
            "next_steps": [],
        }

    # Analyze each dimension
    dimension_results = {}
    for dimension in available_dimensions:
        dim_result = _analyze_dimension_impact(df, dimension, metric_col, min_observations)
        dimension_results[dimension] = dim_result

    # Rank dimensions by total impact
    dimension_rankings = _rank_dimensions_by_impact(dimension_results)

    # Get top segments across all dimensions
    all_segments = []
    for dimension, result in dimension_results.items():
        for segment in result.get("segments", []):
            segment["dimension"] = dimension
            all_segments.append(segment)

    # Sort by absolute contribution
    all_segments.sort(key=lambda x: abs(x.get("contribution", 0)), reverse=True)
    top_segments = all_segments[:top_n]

    # Generate investigation timeline
    timeline = _generate_investigation_timeline(dimension_rankings, top_segments)

    # Create evidence summary
    evidence_summary = _create_evidence_summary(dimension_results, top_segments)

    # Generate next steps
    next_steps = _generate_next_investigation_steps(top_segments, dimension_rankings)

    return {
        "status": "success",
        "dimension_rankings": dimension_rankings,
        "top_segments": top_segments,
        "investigation_timeline": timeline,
        "evidence_summary": evidence_summary,
        "next_steps": next_steps,
        "dimensions_analyzed": len(available_dimensions),
        "total_segments_analyzed": len(all_segments),
    }


def _analyze_dimension_impact(
    df: pd.DataFrame,
    dimension: str,
    metric_col: str,
    min_observations: int,
) -> dict[str, Any]:
    """Analyze impact of a single dimension."""
    # Separate Control and Treatment if available
    has_groups = "group" in df.columns

    if has_groups:
        control_df = df[df["group"] == "Control"]
        treatment_df = df[df["group"] == "Treatment"]

        if control_df.empty or treatment_df.empty:
            has_groups = False

    # Calculate segment-level metrics
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    valid_df = df[metric_values.isin([0, 1])].copy()

    if valid_df.empty:
        return {"segments": [], "total_impact": 0.0, "dimension": dimension}

    segment_stats = valid_df.groupby(dimension).agg(
        conversions=(metric_col, "sum"),
        observations=(metric_col, "count"),
    ).reset_index()

    segment_stats = segment_stats[segment_stats["observations"] >= min_observations]
    segment_stats["rate"] = segment_stats["conversions"] / segment_stats["observations"]

    if has_groups:
        # Calculate contribution relative to Control baseline
        control_rate = control_df[metric_col].mean() if not control_df.empty else 0
        treatment_segments = treatment_df.groupby(dimension).agg(
            conversions=(metric_col, "sum"),
            observations=(metric_col, "count"),
        )
        treatment_segments["rate"] = treatment_segments["conversions"] / treatment_segments["observations"]

        segments = []
        for segment_name in segment_stats[dimension]:
            if segment_name in treatment_segments.index:
                obs = treatment_segments.loc[segment_name, "observations"]
                rate = treatment_segments.loc[segment_name, "rate"]
                expected_conversions = obs * control_rate
                actual_conversions = treatment_segments.loc[segment_name, "conversions"]
                contribution = actual_conversions - expected_conversions

                segments.append({
                    "segment": str(segment_name),
                    "observed_rate": float(rate),
                    "expected_rate": float(control_rate),
                    "observations": int(obs),
                    "contribution": float(contribution),
                    "absolute_change": float(rate - control_rate),
                    "relative_change_pct": float((rate - control_rate) / control_rate * 100) if control_rate > 0 else 0,
                })
    else:
        # Use overall mean as baseline
        overall_rate = segment_stats["rate"].mean()
        segments = []

        for _, row in segment_stats.iterrows():
            rate = row["rate"]
            obs = row["observations"]
            expected_conversions = obs * overall_rate
            actual_conversions = row["conversions"]
            contribution = actual_conversions - expected_conversions

            segments.append({
                "segment": str(row[dimension]),
                "observed_rate": float(rate),
                "expected_rate": float(overall_rate),
                "observations": int(obs),
                "contribution": float(contribution),
                "absolute_change": float(rate - overall_rate),
                "relative_change_pct": float((rate - overall_rate) / overall_rate * 100) if overall_rate > 0 else 0,
            })

    # Sort by absolute contribution
    segments.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Calculate total dimension impact
    total_impact = sum(abs(seg["contribution"]) for seg in segments)

    return {
        "dimension": dimension,
        "segments": segments,
        "total_impact": total_impact,
        "segment_count": len(segments),
    }


def _rank_dimensions_by_impact(dimension_results: dict) -> list[dict]:
    """Rank dimensions by their total impact."""
    rankings = []

    for dimension, result in dimension_results.items():
        total_impact = result.get("total_impact", 0)
        segment_count = result.get("segment_count", 0)

        # Get top contributor in this dimension
        segments = result.get("segments", [])
        top_contributor = segments[0] if segments else None

        rankings.append({
            "dimension": dimension,
            "total_impact": total_impact,
            "segment_count": segment_count,
            "top_segment": top_contributor["segment"] if top_contributor else None,
            "top_segment_contribution": top_contributor["contribution"] if top_contributor else 0,
        })

    rankings.sort(key=lambda x: x["total_impact"], reverse=True)

    return rankings


def _generate_investigation_timeline(
    dimension_rankings: list[dict],
    top_segments: list[dict],
) -> list[dict]:
    """Generate suggested investigation timeline."""
    timeline = []

    # Step 1: Start with highest impact dimension
    if dimension_rankings:
        top_dim = dimension_rankings[0]
        timeline.append({
            "step": 1,
            "action": f"Investigate {top_dim['dimension']} dimension",
            "reason": f"Highest total impact ({top_dim['total_impact']:+.1f} conversions)",
            "focus": f"Start with {top_dim['top_segment']} segment",
            "expected_outcome": "Identify primary driver of change",
        })

    # Step 2: Drill into top segment
    if top_segments:
        top_seg = top_segments[0]
        timeline.append({
            "step": 2,
            "action": f"Deep dive into {top_seg['segment']} ({top_seg['dimension']})",
            "reason": f"Largest single contributor ({top_seg['contribution']:+.1f} conversions)",
            "focus": "Analyze traffic sources, user behavior, and technical metrics",
            "expected_outcome": "Understand specific causes within this segment",
        })

    # Step 3: Compare with second dimension
    if len(dimension_rankings) > 1:
        second_dim = dimension_rankings[1]
        timeline.append({
            "step": 3,
            "action": f"Cross-reference with {second_dim['dimension']}",
            "reason": "Identify interaction effects between dimensions",
            "focus": f"Check if {top_dim['top_segment']} pattern holds across {second_dim['dimension']}",
            "expected_outcome": "Validate findings or discover secondary factors",
        })

    # Step 4: Review time-based patterns
    timeline.append({
        "step": 4,
        "action": "Analyze temporal patterns",
        "reason": "Changes may be time-dependent or event-driven",
        "focus": "Check for day-of-week, time-of-day, or campaign-related patterns",
        "expected_outcome": "Identify timing-based factors",
    })

    # Step 5: Validate and document
    timeline.append({
        "step": 5,
        "action": "Validate findings and document root cause",
        "reason": "Ensure reproducibility and prepare action plan",
        "focus": "Verify data quality, test hypotheses, document evidence",
        "expected_outcome": "Confirmed root cause with action plan",
    })

    return timeline


def _create_evidence_summary(
    dimension_results: dict,
    top_segments: list[dict],
) -> dict[str, Any]:
    """Create aggregated evidence summary."""
    total_segments_analyzed = sum(
        result.get("segment_count", 0) for result in dimension_results.values()
    )

    total_contribution = sum(
        seg["contribution"] for seg in top_segments
    )

    evidence = {
        "dimensions_analyzed": len(dimension_results),
        "total_segments_analyzed": total_segments_analyzed,
        "top_contributors_count": len(top_segments),
        "total_contribution": total_contribution,
        "contribution_concentration": _calculate_concentration(top_segments),
        "evidence_strength": _assess_evidence_strength(dimension_results, top_segments),
    }

    return evidence


def _calculate_concentration(top_segments: list[dict]) -> dict[str, float]:
    """Calculate how concentrated the impact is."""
    if not top_segments:
        return {"top_1": 0, "top_3": 0, "top_5": 0}

    total_contribution = sum(abs(seg["contribution"]) for seg in top_segments)

    if total_contribution == 0:
        return {"top_1": 0, "top_3": 0, "top_5": 0}

    top_1_pct = abs(top_segments[0]["contribution"]) / total_contribution * 100
    top_3_pct = sum(abs(seg["contribution"]) for seg in top_segments[:3]) / total_contribution * 100
    top_5_pct = sum(abs(seg["contribution"]) for seg in top_segments[:5]) / total_contribution * 100

    return {
        "top_1_percentage": top_1_pct,
        "top_3_percentage": top_3_pct,
        "top_5_percentage": top_5_pct,
    }


def _assess_evidence_strength(
    dimension_results: dict,
    top_segments: list[dict],
) -> str:
    """Assess overall evidence strength."""
    if not top_segments:
        return "Insufficient"

    # Check if top segment has large contribution and sufficient observations
    top_seg = top_segments[0]

    if abs(top_seg["contribution"]) > 5 and top_seg["observations"] >= 100:
        return "Strong"
    elif abs(top_seg["contribution"]) > 2 and top_seg["observations"] >= 50:
        return "Moderate"
    else:
        return "Weak"


def _generate_next_investigation_steps(
    top_segments: list[dict],
    dimension_rankings: list[dict],
) -> list[dict]:
    """Generate actionable next steps."""
    steps = []

    if top_segments:
        top_seg = top_segments[0]

        steps.append({
            "priority": "High",
            "action": f"Investigate {top_seg['segment']} in {top_seg['dimension']}",
            "rationale": f"Largest impact: {top_seg['contribution']:+.1f} conversions " +
                        f"({top_seg['relative_change_pct']:+.1f}% change)",
            "specific_checks": [
                f"Review data quality for {top_seg['segment']} records",
                f"Compare {top_seg['segment']} traffic patterns vs baseline",
                f"Check for technical issues affecting {top_seg['segment']}",
                f"Analyze user behavior specific to {top_seg['segment']}",
            ],
        })

    if len(top_segments) > 1:
        second_seg = top_segments[1]

        steps.append({
            "priority": "Medium",
            "action": f"Validate findings with {second_seg['segment']}",
            "rationale": f"Second largest impact: {second_seg['contribution']:+.1f} conversions",
            "specific_checks": [
                f"Check if root cause identified in {top_seg['segment']} applies here",
                "Look for common patterns between top contributors",
            ],
        })

    if dimension_rankings and len(dimension_rankings) > 1:
        steps.append({
            "priority": "Medium",
            "action": "Cross-dimensional analysis",
            "rationale": "Multiple dimensions show impact - may indicate systemic issue",
            "specific_checks": [
                "Create cross-tabs between top 2 dimensions",
                "Check for interaction effects",
                "Identify if certain combinations drive the change",
            ],
        })

    steps.append({
        "priority": "Low",
        "action": "Document and monitor",
        "rationale": "Ensure tracking of identified patterns",
        "specific_checks": [
            "Document all findings with supporting evidence",
            "Set up monitoring for identified segments",
            "Create alerts for future similar patterns",
        ],
    })

    return steps


def drill_down_segment(
    df: pd.DataFrame,
    primary_dimension: str,
    primary_segment: str,
    secondary_dimensions: list[str],
    metric_col: str = "converted",
    min_observations: int = 20,
) -> dict[str, Any]:
    """
    Drill down into a specific segment across secondary dimensions.

    Args:
        df: Input dataframe
        primary_dimension: Main dimension (e.g., 'device')
        primary_segment: Specific segment value (e.g., 'Mobile')
        secondary_dimensions: List of dimensions to analyze within the segment
        metric_col: Metric column
        min_observations: Minimum observations

    Returns:
        Sub-segment analysis results
    """
    # Filter to primary segment
    if primary_dimension not in df.columns:
        return {
            "status": "error",
            "message": f"Dimension {primary_dimension} not found",
        }

    segment_df = df[df[primary_dimension] == primary_segment].copy()

    if segment_df.empty:
        return {
            "status": "error",
            "message": f"No data for {primary_segment} in {primary_dimension}",
        }

    # Analyze each secondary dimension
    sub_analysis = {}
    for dimension in secondary_dimensions:
        if dimension not in segment_df.columns or dimension == primary_dimension:
            continue

        dim_result = _analyze_dimension_impact(
            segment_df, dimension, metric_col, min_observations
        )
        sub_analysis[dimension] = dim_result

    return {
        "status": "success",
        "primary_segment": primary_segment,
        "primary_dimension": primary_dimension,
        "segment_size": len(segment_df),
        "sub_analysis": sub_analysis,
    }
