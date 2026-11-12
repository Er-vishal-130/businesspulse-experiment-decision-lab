"""
Advanced Anomaly Detection Engine

Multiple detection methods with explainability:
- Z-score based detection
- IQR (Interquartile Range) method
- Isolation Forest (when justified)
- Rolling baseline detection
- Change-point detection
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest


def detect_anomalies_multi_method(
    df: pd.DataFrame,
    segment_by: str,
    metric_col: str = "converted",
    methods: list[str] | None = None,
    min_observations: int = 30,
) -> dict[str, Any]:
    """
    Run multiple anomaly detection methods and combine results.

    Args:
        df: Input dataframe
        segment_by: Column to segment by
        metric_col: Metric column to analyze
        methods: List of methods to use ['zscore', 'iqr', 'isolation_forest']
        min_observations: Minimum observations per segment

    Returns:
        Dictionary with results from each method and consensus anomalies
    """
    if methods is None:
        methods = ['zscore', 'iqr']

    if df.empty or segment_by not in df.columns or metric_col not in df.columns:
        return {
            "status": "error",
            "methods_used": [],
            "segment_column": segment_by,
            "anomalies_by_method": {},
            "consensus_anomalies": [],
            "warnings": ["Insufficient data for anomaly detection"],
        }

    results = {
        "status": "success",
        "methods_used": [],
        "segment_column": segment_by,
        "anomalies_by_method": {},
        "consensus_anomalies": [],
        "warnings": [],
    }

    # Run each method
    for method in methods:
        try:
            if method == 'zscore':
                method_result = _detect_zscore_anomalies(df, segment_by, metric_col, min_observations)
            elif method == 'iqr':
                method_result = _detect_iqr_anomalies(df, segment_by, metric_col, min_observations)
            elif method == 'isolation_forest':
                method_result = _detect_isolation_forest_anomalies(df, segment_by, metric_col, min_observations)
            else:
                results["warnings"].append(f"Unsupported anomaly detection method: {method}")
                continue

            results["methods_used"].append(method)
            results["anomalies_by_method"][method] = method_result
            results["warnings"].extend(
                f"{method}: {warning}"
                for warning in method_result.get("warnings", [])
            )
            if method_result.get("reason"):
                results["warnings"].append(f"{method}: {method_result['reason']}")
        except Exception as e:
            results["warnings"].append(f"Method {method} failed: {str(e)}")

    if not results["methods_used"]:
        results["status"] = "error"

    # Find consensus anomalies (detected by multiple methods)
    if len(results["anomalies_by_method"]) > 1:
        results["consensus_anomalies"] = _find_consensus_anomalies(
            results["anomalies_by_method"]
        )
    elif len(results["anomalies_by_method"]) == 1:
        # If only one method, those are the anomalies
        method_name = list(results["anomalies_by_method"].keys())[0]
        results["consensus_anomalies"] = results["anomalies_by_method"][method_name].get("anomalies", [])

    return results


def _detect_zscore_anomalies(
    df: pd.DataFrame,
    segment_by: str,
    metric_col: str,
    min_observations: int,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect anomalies using z-score method."""
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    valid_df = df[metric_values.isin([0, 1])].copy()

    if valid_df.empty:
        return {
            "anomalies": [],
            "method": "Z-score",
            "threshold": threshold,
            "warnings": ["No valid binary metric observations are available"],
        }

    # Calculate conversion rate per segment
    segment_stats = valid_df.groupby(segment_by).agg(
        conversions=(metric_col, "sum"),
        observations=(metric_col, "count"),
    ).reset_index()

    segment_stats = segment_stats[segment_stats["observations"] >= min_observations]

    if len(segment_stats) < 2:
        return {
            "anomalies": [],
            "method": "Z-score",
            "threshold": threshold,
            "warnings": [
                f"At least two segments with {min_observations} observations each are required"
            ],
        }

    segment_stats["rate"] = segment_stats["conversions"] / segment_stats["observations"]

    # Calculate z-scores
    mean_rate = segment_stats["rate"].mean()
    std_rate = segment_stats["rate"].std()

    if std_rate == 0:
        return {
            "anomalies": [],
            "method": "Z-score",
            "threshold": threshold,
            "warnings": ["Segment conversion rates do not vary"],
        }

    segment_stats["z_score"] = (segment_stats["rate"] - mean_rate) / std_rate
    segment_stats["is_anomaly"] = segment_stats["z_score"].abs() > threshold

    anomalies = []
    for _, row in segment_stats[segment_stats["is_anomaly"]].iterrows():
        anomalies.append({
            "segment": str(row[segment_by]),
            "observed_rate": float(row["rate"]),
            "expected_rate": float(mean_rate),
            "z_score": float(row["z_score"]),
            "deviation": float(row["rate"] - mean_rate),
            "severity": _calculate_severity(abs(row["z_score"]), threshold),
            "observations": int(row["observations"]),
        })

    return {
        "anomalies": anomalies,
        "method": "Z-score",
        "threshold": threshold,
        "baseline_mean": float(mean_rate),
        "baseline_std": float(std_rate),
    }


def _detect_iqr_anomalies(
    df: pd.DataFrame,
    segment_by: str,
    metric_col: str,
    min_observations: int,
    multiplier: float = 1.5,
) -> dict[str, Any]:
    """Detect anomalies using IQR method."""
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    valid_df = df[metric_values.isin([0, 1])].copy()

    if valid_df.empty:
        return {
            "anomalies": [],
            "method": "IQR",
            "multiplier": multiplier,
            "warnings": ["No valid binary metric observations are available"],
        }

    # Calculate conversion rate per segment
    segment_stats = valid_df.groupby(segment_by).agg(
        conversions=(metric_col, "sum"),
        observations=(metric_col, "count"),
    ).reset_index()

    segment_stats = segment_stats[segment_stats["observations"] >= min_observations]

    if len(segment_stats) < 4:
        return {
            "anomalies": [],
            "method": "IQR",
            "multiplier": multiplier,
            "warnings": [
                f"At least four segments with {min_observations} observations each are required"
            ],
        }

    segment_stats["rate"] = segment_stats["conversions"] / segment_stats["observations"]

    # Calculate IQR bounds
    q1 = segment_stats["rate"].quantile(0.25)
    q3 = segment_stats["rate"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    segment_stats["is_anomaly"] = (
        (segment_stats["rate"] < lower_bound) | (segment_stats["rate"] > upper_bound)
    )

    anomalies = []
    for _, row in segment_stats[segment_stats["is_anomaly"]].iterrows():
        median_rate = segment_stats["rate"].median()
        anomalies.append({
            "segment": str(row[segment_by]),
            "observed_rate": float(row["rate"]),
            "expected_rate": float(median_rate),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "deviation": float(row["rate"] - median_rate),
            "severity": _calculate_severity_iqr(row["rate"], lower_bound, upper_bound, iqr),
            "observations": int(row["observations"]),
        })

    return {
        "anomalies": anomalies,
        "method": "IQR",
        "multiplier": multiplier,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def _detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    segment_by: str,
    metric_col: str,
    min_observations: int,
    contamination: float = 0.1,
) -> dict[str, Any]:
    """
    Detect anomalies using Isolation Forest (multivariate).
    Use sparingly - only when multiple numeric features exist.
    """
    metric_values = pd.to_numeric(df[metric_col], errors="coerce")
    valid_df = df[metric_values.isin([0, 1])].copy()

    if valid_df.empty:
        return {"anomalies": [], "method": "Isolation Forest", "contamination": contamination}

    # Calculate segment features
    segment_features = valid_df.groupby(segment_by).agg(
        conversions=(metric_col, "sum"),
        observations=(metric_col, "count"),
    )

    # Add additional features if available
    if "revenue" in valid_df.columns:
        revenue_stats = valid_df.groupby(segment_by)["revenue"].agg(["mean", "sum"])
        segment_features = segment_features.join(revenue_stats)

    if "session_duration" in valid_df.columns:
        duration_stats = valid_df.groupby(segment_by)["session_duration"].mean()
        segment_features = segment_features.join(duration_stats.rename("avg_duration"))

    segment_features = segment_features[segment_features["observations"] >= min_observations]

    if len(segment_features) < 10:
        return {
            "anomalies": [],
            "method": "Isolation Forest",
            "contamination": contamination,
            "reason": "Insufficient segments for Isolation Forest (requires ≥10)",
        }

    segment_features["rate"] = segment_features["conversions"] / segment_features["observations"]

    # Select numeric features for model
    feature_cols = [col for col in segment_features.columns if col in ["rate", "mean", "sum", "avg_duration"]]
    X = segment_features[feature_cols].values

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)

    segment_features["is_anomaly"] = predictions == -1

    anomalies = []
    median_rate = segment_features["rate"].median()

    for segment_name, row in segment_features[segment_features["is_anomaly"]].iterrows():
        anomalies.append({
            "segment": str(segment_name),
            "observed_rate": float(row["rate"]),
            "expected_rate": float(median_rate),
            "deviation": float(row["rate"] - median_rate),
            "severity": "Medium",  # Isolation Forest doesn't provide severity scores directly
            "observations": int(row["observations"]),
            "features_used": feature_cols,
        })

    return {
        "anomalies": anomalies,
        "method": "Isolation Forest",
        "contamination": contamination,
        "features_used": feature_cols,
    }


def detect_time_series_anomalies(
    df: pd.DataFrame,
    metric_col: str = "converted",
    date_col: str = "date",
    method: Literal["rolling", "changepoint"] = "rolling",
    window_size: int = 7,
) -> dict[str, Any]:
    """
    Detect anomalies in time-series data.

    Args:
        df: Input dataframe
        metric_col: Metric column
        date_col: Date column
        method: 'rolling' for rolling baseline, 'changepoint' for change detection
        window_size: Rolling window size in periods
    """
    if date_col not in df.columns or metric_col not in df.columns:
        return {
            "anomalies": [],
            "method": method,
            "warnings": ["Required columns missing"],
        }

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col])

    if df_copy.empty:
        return {
            "anomalies": [],
            "method": method,
            "warnings": ["No valid date data"],
        }

    # Aggregate by day
    metric_values = pd.to_numeric(df_copy[metric_col], errors="coerce")
    daily_data = df_copy[metric_values.isin([0, 1])].copy()
    daily_data["date_only"] = daily_data[date_col].dt.date

    daily_stats = daily_data.groupby("date_only").agg(
        conversions=(metric_col, "sum"),
        observations=(metric_col, "count"),
    ).reset_index()

    daily_stats["rate"] = daily_stats["conversions"] / daily_stats["observations"]
    daily_stats = daily_stats.sort_values("date_only")

    if len(daily_stats) < window_size:
        return {
            "anomalies": [],
            "method": method,
            "warnings": [f"Insufficient time periods (need at least {window_size})"],
        }

    if method == "rolling":
        return _detect_rolling_baseline_anomalies(daily_stats, window_size)
    elif method == "changepoint":
        return _detect_changepoint_anomalies(daily_stats)

    return {"anomalies": [], "method": method}


def _detect_rolling_baseline_anomalies(
    daily_stats: pd.DataFrame,
    window_size: int,
) -> dict[str, Any]:
    """Detect anomalies using rolling baseline."""
    daily_stats["rolling_mean"] = daily_stats["rate"].rolling(window=window_size, min_periods=1).mean()
    daily_stats["rolling_std"] = daily_stats["rate"].rolling(window=window_size, min_periods=2).std()

    # Calculate z-score relative to rolling baseline
    daily_stats["z_score"] = (
        (daily_stats["rate"] - daily_stats["rolling_mean"]) / daily_stats["rolling_std"]
    )

    threshold = 2.5
    daily_stats["is_anomaly"] = daily_stats["z_score"].abs() > threshold

    anomalies = []
    for _, row in daily_stats[daily_stats["is_anomaly"]].iterrows():
        if pd.notna(row["z_score"]):
            anomalies.append({
                "date": str(row["date_only"]),
                "observed_rate": float(row["rate"]),
                "expected_rate": float(row["rolling_mean"]),
                "z_score": float(row["z_score"]),
                "deviation": float(row["rate"] - row["rolling_mean"]),
                "severity": _calculate_severity(abs(row["z_score"]), threshold),
                "observations": int(row["observations"]),
            })

    return {
        "anomalies": anomalies,
        "method": "Rolling Baseline",
        "window_size": window_size,
        "threshold": threshold,
    }


def _detect_changepoint_anomalies(daily_stats: pd.DataFrame) -> dict[str, Any]:
    """
    Detect sudden changes in time series (simple implementation).
    Uses day-over-day percentage change.
    """
    daily_stats["pct_change"] = daily_stats["rate"].pct_change() * 100

    # Detect large changes
    threshold_pct = 50  # 50% change
    daily_stats["is_changepoint"] = daily_stats["pct_change"].abs() > threshold_pct

    changepoints = []
    for _, row in daily_stats[daily_stats["is_changepoint"]].iterrows():
        if pd.notna(row["pct_change"]):
            changepoints.append({
                "date": str(row["date_only"]),
                "observed_rate": float(row["rate"]),
                "percentage_change": float(row["pct_change"]),
                "direction": "increase" if row["pct_change"] > 0 else "decrease",
                "severity": "High" if abs(row["pct_change"]) > 75 else "Medium",
                "observations": int(row["observations"]),
            })

    return {
        "anomalies": changepoints,
        "method": "Change-point Detection",
        "threshold_percentage": threshold_pct,
    }


def _find_consensus_anomalies(methods_results: dict) -> list[dict]:
    """Find segments flagged as anomalies by multiple methods."""
    # Collect all segments flagged by each method
    flagged_by_method = {}

    for method_name, result in methods_results.items():
        anomalies = result.get("anomalies", [])
        flagged_segments = {a["segment"] for a in anomalies if "segment" in a}
        flagged_by_method[method_name] = flagged_segments

    if len(flagged_by_method) < 2:
        return []

    # Find segments flagged by at least 2 methods
    all_segments = set()
    for segments in flagged_by_method.values():
        all_segments.update(segments)

    consensus = []
    for segment in all_segments:
        methods_flagging = [
            method for method, segments in flagged_by_method.items()
            if segment in segments
        ]

        if len(methods_flagging) >= 2:
            # Get details from first method that flagged it
            first_method = methods_flagging[0]
            segment_details = next(
                (a for a in methods_results[first_method]["anomalies"] if a.get("segment") == segment),
                None
            )

            if segment_details:
                segment_details["methods_agreeing"] = methods_flagging
                segment_details["consensus_strength"] = len(methods_flagging)
                consensus.append(segment_details)

    return consensus


def _calculate_severity(z_score: float, threshold: float) -> str:
    """Calculate severity based on z-score."""
    if z_score > threshold * 2:
        return "Critical"
    elif z_score > threshold * 1.5:
        return "High"
    elif z_score > threshold:
        return "Medium"
    else:
        return "Low"


def _calculate_severity_iqr(value: float, lower: float, upper: float, iqr: float) -> str:
    """Calculate severity based on IQR distance."""
    if value < lower:
        distance = (lower - value) / iqr if iqr > 0 else 0
    elif value > upper:
        distance = (value - upper) / iqr if iqr > 0 else 0
    else:
        return "Low"

    if distance > 3:
        return "Critical"
    elif distance > 2:
        return "High"
    elif distance > 1:
        return "Medium"
    else:
        return "Low"


def explain_anomaly_detection(method_result: dict) -> str:
    """
    Generate human-readable explanation of detection methodology.
    """
    method = method_result.get("method", "Unknown")

    if method == "Z-score":
        threshold = method_result.get("threshold", 3.0)
        return (
            f"Z-score method: Segments with conversion rates more than {threshold} standard "
            f"deviations from the mean are flagged as anomalies. "
            f"Baseline mean: {method_result.get('baseline_mean', 0):.2%}."
        )

    elif method == "IQR":
        multiplier = method_result.get("multiplier", 1.5)
        return (
            f"IQR method: Segments outside [{method_result.get('lower_bound', 0):.2%}, "
            f"{method_result.get('upper_bound', 0):.2%}] are flagged as anomalies. "
            f"Bounds calculated as Q1/Q3 ± {multiplier} × IQR."
        )

    elif method == "Isolation Forest":
        features = method_result.get("features_used", [])
        return (
            f"Isolation Forest: Machine learning method analyzing multiple features "
            f"({', '.join(features)}) to detect multivariate anomalies. "
            f"Contamination parameter: {method_result.get('contamination', 0.1):.1%}."
        )

    elif method == "Rolling Baseline":
        window = method_result.get("window_size", 7)
        return (
            f"Rolling baseline: Compares each day's performance to a {window}-day rolling average. "
            f"Days with z-score > {method_result.get('threshold', 2.5)} are flagged."
        )

    elif method == "Change-point Detection":
        threshold = method_result.get("threshold_percentage", 50)
        return (
            f"Change-point detection: Identifies sudden shifts in performance. "
            f"Day-over-day changes exceeding {threshold}% are flagged as change points."
        )

    return f"Method: {method}. No detailed explanation available."
