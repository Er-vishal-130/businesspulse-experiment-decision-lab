"""
Cohort & Retention Analytics Module

Provides cohort analysis and customer retention tracking:
- Monthly/weekly cohort creation
- Retention rate calculation
- Cohort comparison
- Repeat behavior analysis
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


def create_cohort_analysis(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    date_col: str = "date",
    cohort_period: str = "M",
    retention_periods: int = 12,
) -> dict[str, Any]:
    """
    Create cohort analysis with retention tracking.

    Args:
        df: Input dataframe
        user_id_col: Column containing user identifiers
        date_col: Column containing dates
        cohort_period: 'M' for monthly, 'W' for weekly cohorts
        retention_periods: Number of periods to track retention

    Returns:
        Dictionary with cohort analysis including:
        - cohort_table: Retention rates by cohort
        - cohort_sizes: Initial size of each cohort
        - retention_curves: Retention over time per cohort
    """
    if user_id_col not in df.columns or date_col not in df.columns:
        return {
            "status": "error",
            "message": f"Required columns missing. Need '{user_id_col}' and '{date_col}'",
        }

    # Prepare data
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col, user_id_col])

    if df_copy.empty:
        return {
            "status": "error",
            "message": "No valid data after filtering",
        }

    # Create cohort assignment (first seen date)
    cohort_data = df_copy.groupby(user_id_col)[date_col].min().reset_index()
    cohort_data.columns = [user_id_col, "cohort_date"]

    # Round to cohort period
    cohort_data["cohort"] = cohort_data["cohort_date"].dt.to_period(cohort_period).dt.start_time

    # Merge back to main data
    df_copy = df_copy.merge(cohort_data[[user_id_col, "cohort"]], on=user_id_col, how="left")

    # Calculate periods since cohort
    df_copy["period"] = df_copy[date_col].dt.to_period(cohort_period).dt.start_time
    df_copy["periods_since_cohort"] = (
        (df_copy["period"].dt.year - df_copy["cohort"].dt.year) * 12 +
        (df_copy["period"].dt.month - df_copy["cohort"].dt.month)
    ) if cohort_period == "M" else (
        ((df_copy["period"] - df_copy["cohort"]).dt.days / 7).astype(int)
    )

    # Create cohort table
    cohort_table = _create_cohort_retention_table(
        df_copy, user_id_col, retention_periods
    )

    # Calculate cohort sizes
    cohort_sizes = df_copy.groupby("cohort")[user_id_col].nunique().reset_index()
    cohort_sizes.columns = ["cohort", "cohort_size"]

    return {
        "status": "success",
        "cohort_table": cohort_table,
        "cohort_sizes": cohort_sizes.to_dict("records"),
        "cohort_period": "Monthly" if cohort_period == "M" else "Weekly",
        "retention_periods_tracked": retention_periods,
    }


def _create_cohort_retention_table(
    df: pd.DataFrame,
    user_id_col: str,
    retention_periods: int,
) -> pd.DataFrame:
    """Create cohort retention table."""
    # Count unique users per cohort per period
    cohort_activity = df.groupby(["cohort", "periods_since_cohort"])[user_id_col].nunique().reset_index()
    cohort_activity.columns = ["cohort", "period", "active_users"]

    # Pivot to create retention table
    retention_table = cohort_activity.pivot(
        index="cohort",
        columns="period",
        values="active_users"
    ).fillna(0)

    # Calculate retention percentages
    retention_pct = retention_table.div(retention_table[0], axis=0) * 100

    # Limit to specified retention periods
    available_periods = min(retention_periods, len(retention_pct.columns))
    retention_pct = retention_pct.iloc[:, :available_periods]

    return retention_pct


def calculate_retention_metrics(cohort_table: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate summary retention metrics.

    Returns:
        Dictionary with key retention metrics
    """
    if cohort_table.empty:
        return {
            "average_retention": {},
            "retention_by_period": {},
        }

    # Average retention by period
    avg_retention_by_period = cohort_table.mean().to_dict()

    # Overall average retention (excluding period 0)
    if len(cohort_table.columns) > 1:
        overall_avg = cohort_table.iloc[:, 1:].mean().mean()
    else:
        overall_avg = 0

    # Identify best and worst performing cohorts
    if len(cohort_table) > 0:
        # Use average retention across all periods
        cohort_avg_retention = cohort_table.mean(axis=1)
        best_cohort = cohort_avg_retention.idxmax()
        worst_cohort = cohort_avg_retention.idxmin()

        best_cohort_retention = cohort_avg_retention.max()
        worst_cohort_retention = cohort_avg_retention.min()
    else:
        best_cohort = None
        worst_cohort = None
        best_cohort_retention = 0
        worst_cohort_retention = 0

    return {
        "overall_average_retention": overall_avg,
        "retention_by_period": avg_retention_by_period,
        "best_cohort": str(best_cohort) if best_cohort else None,
        "best_cohort_retention": best_cohort_retention,
        "worst_cohort": str(worst_cohort) if worst_cohort else None,
        "worst_cohort_retention": worst_cohort_retention,
    }


def analyze_repeat_behavior(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    event_col: str = "converted",
    date_col: str = "date",
) -> dict[str, Any]:
    """
    Analyze repeat purchase or conversion behavior.

    Args:
        df: Input dataframe
        user_id_col: Column with user IDs
        event_col: Column with binary event (1 = converted)
        date_col: Column with dates

    Returns:
        Dictionary with repeat behavior metrics
    """
    if user_id_col not in df.columns or event_col not in df.columns:
        return {
            "status": "error",
            "message": "Required columns missing",
        }

    df_copy = df.copy()
    event_values = pd.to_numeric(df_copy[event_col], errors="coerce")
    df_copy = df_copy[event_values == 1].copy()

    if df_copy.empty:
        return {
            "status": "error",
            "message": "No conversion events found",
        }

    # Count conversions per user
    conversion_counts = df_copy.groupby(user_id_col).size().reset_index(name="conversion_count")

    # Calculate repeat metrics
    total_users = len(conversion_counts)
    one_time_users = (conversion_counts["conversion_count"] == 1).sum()
    repeat_users = total_users - one_time_users

    repeat_rate = repeat_users / total_users if total_users > 0 else 0

    # Distribution of conversion counts
    conversion_distribution = conversion_counts["conversion_count"].value_counts().sort_index()

    # Average conversions per user
    avg_conversions = conversion_counts["conversion_count"].mean()

    # Time between conversions
    if date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy = df_copy.dropna(subset=[date_col])
        df_copy = df_copy.sort_values([user_id_col, date_col])

        df_copy["next_conversion_date"] = df_copy.groupby(user_id_col)[date_col].shift(-1)
        df_copy["days_to_next"] = (df_copy["next_conversion_date"] - df_copy[date_col]).dt.days

        avg_days_between = df_copy["days_to_next"].mean()
    else:
        avg_days_between = None

    return {
        "status": "success",
        "total_converting_users": int(total_users),
        "one_time_users": int(one_time_users),
        "repeat_users": int(repeat_users),
        "repeat_rate": repeat_rate,
        "average_conversions_per_user": avg_conversions,
        "conversion_distribution": conversion_distribution.to_dict(),
        "average_days_between_conversions": avg_days_between,
    }


def compare_cohort_performance(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    date_col: str = "date",
    metric_col: str = "revenue",
    cohort_period: str = "M",
) -> dict[str, Any]:
    """
    Compare performance metrics across cohorts.

    Args:
        df: Input dataframe
        user_id_col: User ID column
        date_col: Date column
        metric_col: Metric to compare (e.g., 'revenue')
        cohort_period: 'M' or 'W'

    Returns:
        Cohort performance comparison
    """
    if user_id_col not in df.columns or date_col not in df.columns:
        return {
            "status": "error",
            "message": "Required columns missing",
        }

    if metric_col not in df.columns:
        return {
            "status": "error",
            "message": f"Metric column '{metric_col}' not found",
        }

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col, user_id_col])

    # Assign cohorts
    cohort_data = df_copy.groupby(user_id_col)[date_col].min().reset_index()
    cohort_data.columns = [user_id_col, "cohort_date"]
    cohort_data["cohort"] = cohort_data["cohort_date"].dt.to_period(cohort_period).dt.start_time

    df_copy = df_copy.merge(cohort_data[[user_id_col, "cohort"]], on=user_id_col, how="left")

    # Calculate metrics per cohort
    metric_values = pd.to_numeric(df_copy[metric_col], errors="coerce")
    df_copy[metric_col] = metric_values

    cohort_metrics = df_copy.groupby("cohort").agg(
        total_metric=(metric_col, "sum"),
        users=(user_id_col, "nunique"),
    ).reset_index()

    cohort_metrics["metric_per_user"] = cohort_metrics["total_metric"] / cohort_metrics["users"]

    return {
        "status": "success",
        "cohort_metrics": cohort_metrics.to_dict("records"),
        "metric_name": metric_col,
        "best_cohort": str(cohort_metrics.loc[cohort_metrics["metric_per_user"].idxmax(), "cohort"]),
        "worst_cohort": str(cohort_metrics.loc[cohort_metrics["metric_per_user"].idxmin(), "cohort"]),
    }
