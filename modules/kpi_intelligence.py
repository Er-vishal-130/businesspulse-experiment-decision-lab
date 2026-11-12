"""
KPI Intelligence Module

Provides advanced KPI tracking, trend analysis, and performance monitoring:
- Automatic KPI detection from dataset
- Historical trend analysis
- Period-over-period comparisons
- Target vs actual tracking
- KPI anomaly detection
- Performance scoring
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def detect_available_kpis(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Automatically detect which KPIs can be calculated from the dataset.

    Returns:
        Dictionary mapping KPI name to metadata including:
        - available: bool
        - required_columns: list
        - missing_columns: list
        - description: str
    """
    kpi_definitions = {
        "Total Revenue": {
            "required_columns": ["revenue"],
            "description": "Sum of all revenue",
            "calculation_type": "sum",
        },
        "Conversion Rate": {
            "required_columns": ["converted"],
            "description": "Percentage of users who converted",
            "calculation_type": "rate",
        },
        "Average Order Value": {
            "required_columns": ["revenue", "converted"],
            "description": "Average revenue per conversion",
            "calculation_type": "average",
        },
        "Total Conversions": {
            "required_columns": ["converted"],
            "description": "Total number of conversions",
            "calculation_type": "sum",
        },
        "Total Users": {
            "required_columns": ["user_id"],
            "description": "Count of unique users",
            "calculation_type": "count_unique",
        },
        "Total Sessions": {
            "required_columns": ["session_id"],
            "description": "Count of unique sessions",
            "calculation_type": "count_unique",
        },
        "Bounce Rate": {
            "required_columns": ["bounce"],
            "description": "Percentage of single-page sessions",
            "calculation_type": "rate",
        },
        "Average Session Duration": {
            "required_columns": ["session_duration"],
            "description": "Average time spent per session",
            "calculation_type": "average",
        },
        "Average Pages Viewed": {
            "required_columns": ["pages_viewed"],
            "description": "Average pages viewed per session",
            "calculation_type": "average",
        },
        "Revenue Per User": {
            "required_columns": ["revenue", "user_id"],
            "description": "Average revenue per unique user",
            "calculation_type": "average",
        },
        "Error Rate": {
            "required_columns": ["error_flag"],
            "description": "Percentage of sessions with errors",
            "calculation_type": "rate",
        },
    }

    available_kpis = {}

    for kpi_name, definition in kpi_definitions.items():
        required = definition["required_columns"]
        available = all(col in df.columns for col in required)
        missing = [col for col in required if col not in df.columns]

        available_kpis[kpi_name] = {
            "available": available,
            "required_columns": required,
            "missing_columns": missing,
            "description": definition["description"],
            "calculation_type": definition["calculation_type"],
        }

    return available_kpis


def calculate_kpi_values(df: pd.DataFrame, kpi_availability: dict) -> dict[str, Any]:
    """
    Calculate all available KPI values.

    Returns:
        Dictionary mapping KPI name to calculated value
    """
    kpi_values = {}

    for kpi_name, metadata in kpi_availability.items():
        if not metadata["available"]:
            kpi_values[kpi_name] = {
                "value": None,
                "formatted_value": "N/A",
                "status": "unavailable",
                "reason": f"Missing columns: {', '.join(metadata['missing_columns'])}",
            }
            continue

        try:
            value = _calculate_single_kpi(df, kpi_name, metadata["calculation_type"])
            kpi_values[kpi_name] = {
                "value": value,
                "formatted_value": _format_kpi_value(kpi_name, value),
                "status": "available",
                "reason": None,
            }
        except Exception as e:
            kpi_values[kpi_name] = {
                "value": None,
                "formatted_value": "Error",
                "status": "error",
                "reason": str(e),
            }

    return kpi_values


def _calculate_single_kpi(df: pd.DataFrame, kpi_name: str, calc_type: str) -> float:
    """Calculate a single KPI value."""
    if kpi_name == "Total Revenue":
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        return float(revenue.sum())

    elif kpi_name == "Conversion Rate":
        converted = pd.to_numeric(df["converted"], errors="coerce")
        valid = converted[converted.isin([0, 1])]
        if len(valid) == 0:
            return 0.0
        return float(valid.mean())

    elif kpi_name == "Average Order Value":
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        converted = pd.to_numeric(df["converted"], errors="coerce")
        valid_conversions = converted[converted == 1]
        if len(valid_conversions) == 0:
            return 0.0
        revenue_from_conversions = revenue[converted == 1]
        return float(revenue_from_conversions.mean())

    elif kpi_name == "Total Conversions":
        converted = pd.to_numeric(df["converted"], errors="coerce")
        return float(converted.sum())

    elif kpi_name == "Total Users":
        if "user_id" in df.columns:
            return float(df["user_id"].nunique())
        return float(len(df))

    elif kpi_name == "Total Sessions":
        if "session_id" in df.columns:
            return float(df["session_id"].nunique())
        return float(len(df))

    elif kpi_name == "Bounce Rate":
        bounce = pd.to_numeric(df["bounce"], errors="coerce")
        valid = bounce[bounce.isin([0, 1])]
        if len(valid) == 0:
            return 0.0
        return float(valid.mean())

    elif kpi_name == "Average Session Duration":
        duration = pd.to_numeric(df["session_duration"], errors="coerce")
        return float(duration.mean())

    elif kpi_name == "Average Pages Viewed":
        pages = pd.to_numeric(df["pages_viewed"], errors="coerce")
        return float(pages.mean())

    elif kpi_name == "Revenue Per User":
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        unique_users = df["user_id"].nunique() if "user_id" in df.columns else len(df)
        if unique_users == 0:
            return 0.0
        return float(revenue.sum() / unique_users)

    elif kpi_name == "Error Rate":
        error_flag = pd.to_numeric(df["error_flag"], errors="coerce")
        valid = error_flag[error_flag.isin([0, 1])]
        if len(valid) == 0:
            return 0.0
        return float(valid.mean())

    return 0.0


def _format_kpi_value(kpi_name: str, value: float) -> str:
    """Format KPI value for display."""
    if value is None or np.isnan(value):
        return "N/A"

    if "Rate" in kpi_name or "Percentage" in kpi_name:
        return f"{value * 100:.2f}%"
    elif "Revenue" in kpi_name or "Value" in kpi_name:
        return f"${value:,.2f}"
    elif "Duration" in kpi_name:
        return f"{value:.1f} sec"
    elif value >= 1000:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"


def calculate_kpi_trends(
    df: pd.DataFrame,
    kpi_availability: dict,
    date_column: str = "date",
    frequency: str = "D",
) -> dict[str, pd.DataFrame]:
    """
    Calculate KPI trends over time.

    Args:
        df: Input dataframe
        kpi_availability: KPI availability metadata
        date_column: Name of date column
        frequency: Pandas frequency string ('D', 'W', 'M')

    Returns:
        Dictionary mapping KPI name to trend dataframe
    """
    if date_column not in df.columns:
        return {}

    trends = {}

    # Convert date column
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_column])

    if df_copy.empty:
        return {}

    # Group by date
    df_copy["date_period"] = df_copy[date_column].dt.to_period(frequency).dt.start_time

    for kpi_name, metadata in kpi_availability.items():
        if not metadata["available"]:
            continue

        try:
            trend_data = _calculate_kpi_trend(df_copy, kpi_name, metadata["calculation_type"])
            if not trend_data.empty:
                trends[kpi_name] = trend_data
        except Exception:
            continue

    return trends


def _calculate_kpi_trend(df: pd.DataFrame, kpi_name: str, calc_type: str) -> pd.DataFrame:
    """Calculate trend for a single KPI."""
    if kpi_name == "Conversion Rate":
        converted = pd.to_numeric(df["converted"], errors="coerce")
        valid_df = df[converted.isin([0, 1])].copy()
        trend = valid_df.groupby("date_period")["converted"].agg(
            conversions="sum",
            total="count",
        )
        trend["value"] = trend["conversions"] / trend["total"]
        return trend[["value"]].reset_index()

    elif kpi_name == "Total Revenue":
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        trend = df.groupby("date_period").agg(value=("revenue", "sum"))
        return trend.reset_index()

    elif kpi_name == "Bounce Rate":
        bounce = pd.to_numeric(df["bounce"], errors="coerce")
        valid_df = df[bounce.isin([0, 1])].copy()
        trend = valid_df.groupby("date_period")["bounce"].mean()
        return trend.to_frame("value").reset_index()

    # Add more KPI-specific calculations as needed

    return pd.DataFrame()


def compare_kpi_periods(
    df: pd.DataFrame,
    kpi_values: dict,
    date_column: str = "date",
    period_1_label: str = "Previous",
    period_2_label: str = "Current",
) -> dict[str, dict]:
    """
    Compare KPIs between two time periods.

    Returns:
        Dictionary with period comparisons for each KPI
    """
    if date_column not in df.columns:
        return {}

    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_column])

    if df_copy.empty:
        return {}

    # Split into two equal periods
    min_date = df_copy[date_column].min()
    max_date = df_copy[date_column].max()
    mid_date = min_date + (max_date - min_date) / 2

    period_1 = df_copy[df_copy[date_column] < mid_date]
    period_2 = df_copy[df_copy[date_column] >= mid_date]

    comparisons = {}

    for kpi_name, kpi_info in kpi_values.items():
        if kpi_info["status"] != "available":
            continue

        try:
            val_1 = _calculate_single_kpi(period_1, kpi_name, "sum")
            val_2 = _calculate_single_kpi(period_2, kpi_name, "sum")

            change = val_2 - val_1
            pct_change = (change / val_1 * 100) if val_1 != 0 else 0.0

            comparisons[kpi_name] = {
                f"{period_1_label}_value": val_1,
                f"{period_2_label}_value": val_2,
                "absolute_change": change,
                "percentage_change": pct_change,
                "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
            }
        except Exception:
            continue

    return comparisons


def detect_kpi_anomalies(
    trends: dict[str, pd.DataFrame],
    method: str = "zscore",
    threshold: float = 3.0,
) -> dict[str, list[dict]]:
    """
    Detect anomalies in KPI time series.

    Args:
        trends: Dictionary of KPI trends (from calculate_kpi_trends)
        method: Detection method ('zscore', 'iqr', 'rolling')
        threshold: Threshold for anomaly detection

    Returns:
        Dictionary mapping KPI name to list of anomalies
    """
    anomalies = {}

    for kpi_name, trend_df in trends.items():
        if trend_df.empty or "value" not in trend_df.columns:
            continue

        kpi_anomalies = []

        if method == "zscore":
            mean = trend_df["value"].mean()
            std = trend_df["value"].std()

            if std == 0:
                continue

            z_scores = np.abs((trend_df["value"] - mean) / std)
            anomaly_mask = z_scores > threshold

            for idx in anomaly_mask[anomaly_mask].index:
                kpi_anomalies.append({
                    "date": str(trend_df.loc[idx, "date_period"]),
                    "value": float(trend_df.loc[idx, "value"]),
                    "z_score": float(z_scores[idx]),
                    "expected": float(mean),
                    "deviation": float(trend_df.loc[idx, "value"] - mean),
                })

        elif method == "iqr":
            q1 = trend_df["value"].quantile(0.25)
            q3 = trend_df["value"].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            anomaly_mask = (trend_df["value"] < lower_bound) | (trend_df["value"] > upper_bound)

            for idx in anomaly_mask[anomaly_mask].index:
                kpi_anomalies.append({
                    "date": str(trend_df.loc[idx, "date_period"]),
                    "value": float(trend_df.loc[idx, "value"]),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                })

        if kpi_anomalies:
            anomalies[kpi_name] = kpi_anomalies

    return anomalies


def calculate_kpi_health_score(kpi_values: dict, targets: dict[str, float] = None) -> dict[str, Any]:
    """
    Calculate overall KPI health score.

    Args:
        kpi_values: Calculated KPI values
        targets: Optional dictionary of target values per KPI

    Returns:
        Dictionary with health score and component scores
    """
    available_count = sum(1 for kpi in kpi_values.values() if kpi["status"] == "available")
    total_count = len(kpi_values)

    availability_score = (available_count / total_count * 100) if total_count > 0 else 0

    # If targets provided, calculate target achievement
    target_score = 100.0
    if targets:
        achieved = 0
        total_with_targets = 0

        for kpi_name, target_value in targets.items():
            if kpi_name in kpi_values and kpi_values[kpi_name]["status"] == "available":
                actual_value = kpi_values[kpi_name]["value"]
                if actual_value >= target_value:
                    achieved += 1
                total_with_targets += 1

        if total_with_targets > 0:
            target_score = (achieved / total_with_targets * 100)

    overall_score = (availability_score * 0.4 + target_score * 0.6)

    return {
        "overall_score": round(overall_score, 2),
        "availability_score": round(availability_score, 2),
        "target_achievement_score": round(target_score, 2),
        "available_kpis": available_count,
        "total_kpis": total_count,
        "health_grade": _get_health_grade(overall_score),
    }


def _get_health_grade(score: float) -> str:
    """Convert health score to grade."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Poor"
