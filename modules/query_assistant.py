"""
Enhanced Query Assistant Module

Improved natural language query understanding with:
- Extended query patterns
- Context-aware responses
- Calculation methodology explanation
- Confidence scoring
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


def answer_query_enhanced(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """
    Enhanced query answering with better understanding and explanations.

    Args:
        df: Input dataframe
        query: Natural language query

    Returns:
        Dictionary with:
        - answer: Text response
        - data: Supporting data (if applicable)
        - confidence: Confidence score (0-100)
        - methodology: Explanation of how answer was calculated
        - suggestions: Follow-up query suggestions
    """
    query_lower = query.lower().strip()

    # Try to match query patterns
    patterns = [
        (_pattern_which_segment_highest, ["which", "segment", "highest", "best"]),
        (_pattern_which_segment_lowest, ["which", "segment", "lowest", "worst"]),
        (_pattern_conversion_comparison, ["treatment", "control", "compare", "vs"]),
        (_pattern_revenue_analysis, ["revenue", "total", "sum"]),
        (_pattern_device_performance, ["device", "performance"]),
        (_pattern_source_performance, ["traffic", "source", "performance"]),
        (_pattern_anomaly_query, ["anomaly", "unusual", "outlier"]),
        (_pattern_trend_query, ["trend", "over time", "daily", "weekly"]),
        (_pattern_kpi_status, ["kpi", "metrics", "performance"]),
        (_pattern_data_quality, ["data quality", "missing", "duplicates"]),
        (_pattern_top_n, ["top", "bottom", "best", "worst"]),
    ]

    for pattern_func, keywords in patterns:
        if any(keyword in query_lower for keyword in keywords):
            result = pattern_func(df, query_lower)
            if result["confidence"] > 50:
                return result

    # Default fallback
    return {
        "answer": "I couldn't understand that query. Try asking about:\n" +
                 "• Which segment has the highest/lowest conversion rate?\n" +
                 "• How does Treatment compare to Control?\n" +
                 "• What is the total revenue?\n" +
                 "• Which device performs best?\n" +
                 "• What are the top 5 segments by conversion?",
        "data": None,
        "confidence": 0,
        "methodology": "Query not matched to known patterns",
        "suggestions": [
            "Which device has the highest conversion rate?",
            "Compare treatment and control conversion rates",
            "What is the total revenue?",
        ],
    }


def _pattern_which_segment_highest(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle 'which segment has highest X' queries."""
    # Determine dimension
    dimension = None
    if "device" in query:
        dimension = "device"
    elif "traffic" in query or "source" in query:
        dimension = "traffic_source"
    elif "country" in query:
        dimension = "country"
    elif "gender" in query:
        dimension = "gender"

    if dimension and dimension not in df.columns:
        return {"answer": f"Column '{dimension}' not found in dataset", "confidence": 0}

    if not dimension:
        # Try to infer from available columns
        candidate_dims = ["device", "traffic_source", "country", "gender"]
        dimension = next((d for d in candidate_dims if d in df.columns), None)

    if not dimension:
        return {"answer": "No segmentation dimension found", "confidence": 0}

    # Calculate conversion by segment
    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    if valid_df.empty:
        return {"answer": "No valid conversion data", "confidence": 0}

    segment_stats = valid_df.groupby(dimension).agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    segment_stats["conversion_rate"] = segment_stats["conversions"] / segment_stats["observations"]

    best_segment = segment_stats["conversion_rate"].idxmax()
    best_rate = segment_stats.loc[best_segment, "conversion_rate"]
    best_obs = segment_stats.loc[best_segment, "observations"]

    answer = (
        f"**{best_segment}** has the highest conversion rate in the {dimension} dimension: "
        f"**{best_rate:.2%}** (based on {best_obs:,} observations)"
    )

    # Create comparison table
    comparison_table = segment_stats.sort_values("conversion_rate", ascending=False).reset_index()
    comparison_table["conversion_rate"] = comparison_table["conversion_rate"].apply(lambda x: f"{x:.2%}")

    return {
        "answer": answer,
        "data": comparison_table,
        "confidence": 90,
        "methodology": f"Calculated conversion rate (conversions ÷ observations) for each {dimension} segment. " +
                      f"Minimum observation requirement: {best_obs >= 30}.",
        "suggestions": [
            f"Which {dimension} has the lowest conversion rate?",
            f"Compare {dimension} performance by group",
            "What is the overall conversion rate?",
        ],
    }


def _pattern_which_segment_lowest(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle 'which segment has lowest X' queries."""
    dimension = None
    if "device" in query:
        dimension = "device"
    elif "traffic" in query or "source" in query:
        dimension = "traffic_source"
    elif "country" in query:
        dimension = "country"

    if not dimension or dimension not in df.columns:
        return {"answer": "Dimension not identified or not available", "confidence": 0}

    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    segment_stats = valid_df.groupby(dimension).agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    segment_stats["conversion_rate"] = segment_stats["conversions"] / segment_stats["observations"]

    worst_segment = segment_stats["conversion_rate"].idxmin()
    worst_rate = segment_stats.loc[worst_segment, "conversion_rate"]
    worst_obs = segment_stats.loc[worst_segment, "observations"]

    answer = (
        f"**{worst_segment}** has the lowest conversion rate in {dimension}: "
        f"**{worst_rate:.2%}** ({worst_obs:,} observations)"
    )

    comparison_table = segment_stats.sort_values("conversion_rate").reset_index()
    comparison_table["conversion_rate"] = comparison_table["conversion_rate"].apply(lambda x: f"{x:.2%}")

    return {
        "answer": answer,
        "data": comparison_table,
        "confidence": 90,
        "methodology": f"Identified segment with lowest conversion rate in {dimension} dimension",
        "suggestions": [
            f"Investigate why {worst_segment} underperforms",
            f"Compare {worst_segment} with best performing segment",
        ],
    }


def _pattern_conversion_comparison(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle treatment vs control comparison queries."""
    if "group" not in df.columns:
        return {"answer": "No group column found for comparison", "confidence": 0}

    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    group_stats = valid_df.groupby("group").agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    group_stats["conversion_rate"] = group_stats["conversions"] / group_stats["observations"]

    if "Control" not in group_stats.index or "Treatment" not in group_stats.index:
        return {"answer": "Both Control and Treatment groups required", "confidence": 0}

    ctrl_rate = group_stats.loc["Control", "conversion_rate"]
    trt_rate = group_stats.loc["Treatment", "conversion_rate"]
    diff = trt_rate - ctrl_rate
    pct_change = (diff / ctrl_rate * 100) if ctrl_rate > 0 else 0

    if diff > 0:
        answer = (
            f"**Treatment performs better** than Control:\n"
            f"• Treatment: **{trt_rate:.2%}**\n"
            f"• Control: **{ctrl_rate:.2%}**\n"
            f"• Absolute lift: **{diff:+.2%}**\n"
            f"• Relative lift: **{pct_change:+.1f}%**"
        )
    elif diff < 0:
        answer = (
            f"**Control performs better** than Treatment:\n"
            f"• Control: **{ctrl_rate:.2%}**\n"
            f"• Treatment: **{trt_rate:.2%}**\n"
            f"• Absolute difference: **{diff:.2%}**\n"
            f"• Relative change: **{pct_change:.1f}%**"
        )
    else:
        answer = "Treatment and Control have **identical conversion rates**"

    return {
        "answer": answer,
        "data": group_stats.reset_index(),
        "confidence": 95,
        "methodology": "Direct comparison of conversion rates between Control and Treatment groups. " +
                      "Statistical significance requires hypothesis testing (see Statistics page).",
        "suggestions": [
            "Run statistical test for significance",
            "Compare segments within Treatment",
            "Check guardrail metrics",
        ],
    }


def _pattern_revenue_analysis(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle revenue-related queries."""
    if "revenue" not in df.columns:
        return {"answer": "No revenue column found in dataset", "confidence": 0}

    revenue = pd.to_numeric(df["revenue"], errors="coerce")
    total_revenue = revenue.sum()

    if "group" in df.columns and ("by group" in query or "per group" in query):
        group_revenue = df.groupby("group")["revenue"].sum().reset_index()
        answer = f"**Total revenue: ${total_revenue:,.2f}**\n\nBy group:\n"
        for _, row in group_revenue.iterrows():
            answer += f"• {row['group']}: ${row['revenue']:,.2f}\n"

        return {
            "answer": answer,
            "data": group_revenue,
            "confidence": 95,
            "methodology": "Summed all revenue values, then grouped by experiment group",
            "suggestions": ["Calculate revenue per user", "Compare revenue by device"],
        }

    answer = f"**Total revenue: ${total_revenue:,.2f}**"

    return {
        "answer": answer,
        "data": None,
        "confidence": 95,
        "methodology": "Summed all revenue column values",
        "suggestions": ["Show revenue by group", "Calculate revenue per conversion"],
    }


def _pattern_device_performance(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle device performance queries."""
    if "device" not in df.columns:
        return {"answer": "No device column in dataset", "confidence": 0}

    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    device_stats = valid_df.groupby("device").agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    device_stats["conversion_rate"] = device_stats["conversions"] / device_stats["observations"]
    device_stats = device_stats.sort_values("conversion_rate", ascending=False)

    answer = "**Device Performance (by conversion rate):**\n\n"
    for device, row in device_stats.iterrows():
        answer += f"• {device}: **{row['conversion_rate']:.2%}** ({row['observations']:,} users)\n"

    return {
        "answer": answer,
        "data": device_stats.reset_index(),
        "confidence": 90,
        "methodology": "Calculated conversion rate per device type",
        "suggestions": ["Which device improved most in Treatment?", "Compare mobile vs desktop"],
    }


def _pattern_source_performance(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle traffic source performance."""
    if "traffic_source" not in df.columns:
        return {"answer": "No traffic_source column found", "confidence": 0}

    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    source_stats = valid_df.groupby("traffic_source").agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    source_stats["conversion_rate"] = source_stats["conversions"] / source_stats["observations"]
    source_stats = source_stats.sort_values("conversion_rate", ascending=False)

    answer = "**Traffic Source Performance:**\n\n"
    for source, row in source_stats.iterrows():
        answer += f"• {source}: **{row['conversion_rate']:.2%}** ({row['observations']:,} users)\n"

    return {
        "answer": answer,
        "data": source_stats.reset_index(),
        "confidence": 90,
        "methodology": "Calculated conversion rate per traffic source",
        "suggestions": ["Which source has best ROI?", "Compare paid vs organic"],
    }


def _pattern_anomaly_query(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle anomaly detection queries."""
    return {
        "answer": "For anomaly detection, please use the **BusinessPulse Detective** page or **Anomaly Detection** page. " +
                 "These provide statistical anomaly identification across segments.",
        "data": None,
        "confidence": 70,
        "methodology": "Anomaly detection requires dedicated statistical analysis",
        "suggestions": ["Go to BusinessPulse Detective page", "Check Alert Center"],
    }


def _pattern_trend_query(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle trend analysis queries."""
    if "date" not in df.columns:
        return {"answer": "No date column found for trend analysis", "confidence": 0}

    return {
        "answer": "For trend analysis, view the **Analytics** page or **Advanced Charts** > Trend Ribbon. " +
                 "These show conversion rates over time.",
        "data": None,
        "confidence": 70,
        "methodology": "Time-series analysis available in visualization pages",
        "suggestions": ["Go to Analytics page", "Check Advanced Charts"],
    }


def _pattern_kpi_status(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle KPI status queries."""
    return {
        "answer": "For comprehensive KPI tracking, visit the **KPI Intelligence** page. " +
                 "This shows all available metrics with trends and comparisons.",
        "data": None,
        "confidence": 75,
        "methodology": "KPI dashboard provides detailed metric analysis",
        "suggestions": ["Go to KPI Intelligence page", "Check Overview page"],
    }


def _pattern_data_quality(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle data quality queries."""
    return {
        "answer": "For data quality assessment, visit the **Data Quality Center** page. " +
                 "This provides comprehensive profiling, quality scoring, and issue detection.",
        "data": None,
        "confidence": 75,
        "methodology": "Data Quality Center provides detailed profiling",
        "suggestions": ["Go to Data Quality Center", "Check Data Validation page"],
    }


def _pattern_top_n(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Handle 'top N' queries."""
    # Extract number if present
    import re
    numbers = re.findall(r'\d+', query)
    n = int(numbers[0]) if numbers else 5

    # Determine dimension and metric
    converted = pd.to_numeric(df["converted"], errors="coerce")
    valid_df = df[converted.isin([0, 1])].copy()

    # Try common dimensions
    dimension = None
    for dim in ["device", "traffic_source", "country", "gender"]:
        if dim in query and dim in df.columns:
            dimension = dim
            break

    if not dimension:
        # Use first available
        for dim in ["device", "traffic_source", "country"]:
            if dim in df.columns:
                dimension = dim
                break

    if not dimension:
        return {"answer": "No suitable dimension found for ranking", "confidence": 0}

    segment_stats = valid_df.groupby(dimension).agg(
        conversions=("converted", "sum"),
        observations=("converted", "count"),
    )
    segment_stats["conversion_rate"] = segment_stats["conversions"] / segment_stats["observations"]

    if "bottom" in query or "worst" in query:
        top_segments = segment_stats.nsmallest(n, "conversion_rate")
        answer = f"**Bottom {n} {dimension} segments by conversion rate:**\n\n"
    else:
        top_segments = segment_stats.nlargest(n, "conversion_rate")
        answer = f"**Top {n} {dimension} segments by conversion rate:**\n\n"

    for segment, row in top_segments.iterrows():
        answer += f"• {segment}: **{row['conversion_rate']:.2%}** ({row['observations']:,} users)\n"

    return {
        "answer": answer,
        "data": top_segments.reset_index(),
        "confidence": 85,
        "methodology": f"Ranked all {dimension} segments by conversion rate and selected top {n}",
        "suggestions": [f"Compare top vs bottom {dimension}", f"Investigate best {dimension}"],
    }
