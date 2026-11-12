"""
Data Quality & Profiling Engine

Provides comprehensive data quality assessment including:
- Column-level profiling
- Data quality scoring
- Outlier detection
- Missing value analysis
- Duplicate detection
- Inconsistent category detection
- Data completeness metrics
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Comprehensive data profiling with quality assessment.

    Returns:
        Dictionary containing:
        - overall_quality_score: 0-100 score
        - total_rows: int
        - total_columns: int
        - column_profiles: list of column-level profiles
        - missing_analysis: missing value summary
        - duplicate_analysis: duplicate row summary
        - data_types: inferred types per column
        - issues: list of detected issues
        - recommendations: list of suggested fixes
    """
    if df is None or df.empty:
        return _empty_profile("No data available for profiling")

    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_profiles": [],
        "missing_analysis": {},
        "duplicate_analysis": {},
        "data_types": {},
        "issues": [],
        "recommendations": [],
        "overall_quality_score": 0.0,
    }

    # Column-level profiling
    for column in df.columns:
        col_profile = _profile_column(df[column], column)
        profile["column_profiles"].append(col_profile)
        profile["data_types"][column] = col_profile["inferred_type"]

        # Collect issues
        if col_profile["has_issues"]:
            profile["issues"].extend(col_profile["issues"])
            profile["recommendations"].extend(col_profile["recommendations"])

    # Missing value analysis
    profile["missing_analysis"] = _analyze_missing_values(df)

    # Duplicate analysis
    profile["duplicate_analysis"] = _analyze_duplicates(df)

    # Calculate overall quality score
    profile["overall_quality_score"] = _calculate_quality_score(
        df, profile["column_profiles"], profile["missing_analysis"],
        profile["duplicate_analysis"]
    )

    return profile


def _empty_profile(reason: str) -> dict[str, Any]:
    """Return empty profile structure with reason."""
    return {
        "total_rows": 0,
        "total_columns": 0,
        "column_profiles": [],
        "missing_analysis": {},
        "duplicate_analysis": {},
        "data_types": {},
        "issues": [reason],
        "recommendations": [],
        "overall_quality_score": 0.0,
    }


def _profile_column(series: pd.Series, column_name: str) -> dict[str, Any]:
    """
    Profile a single column with statistics and quality checks.
    """
    profile = {
        "column_name": column_name,
        "inferred_type": str(series.dtype),
        "non_null_count": int(series.notna().sum()),
        "null_count": int(series.isna().sum()),
        "null_percentage": float(series.isna().sum() / len(series) * 100) if len(series) > 0 else 0.0,
        "unique_count": int(series.nunique(dropna=False)),
        "unique_percentage": float(series.nunique(dropna=False) / len(series) * 100) if len(series) > 0 else 0.0,
        "has_issues": False,
        "issues": [],
        "recommendations": [],
        "statistics": {},
        "outliers": None,
    }

    # Check if column is constant
    if profile["unique_count"] == 1:
        profile["has_issues"] = True
        profile["issues"].append(f"Column '{column_name}' has constant value")
        profile["recommendations"].append(f"Consider removing constant column '{column_name}' (no information gain)")

    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(series):
        profile["statistics"] = _numeric_statistics(series)
        profile["outliers"] = _detect_numeric_outliers(series)

        if profile["outliers"]["count"] > 0:
            outlier_pct = profile["outliers"]["percentage"]
            if outlier_pct > 10:
                profile["has_issues"] = True
                profile["issues"].append(f"Column '{column_name}' has {outlier_pct:.1f}% outliers")
                profile["recommendations"].append(
                    f"Review outliers in '{column_name}' - they may indicate data quality issues or genuine extremes"
                )

    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        profile["statistics"] = _categorical_statistics(series)

        # Check for inconsistent categories (e.g., "Mobile" vs "mobile" vs " Mobile ")
        inconsistencies = _detect_category_inconsistencies(series)
        if inconsistencies:
            profile["has_issues"] = True
            profile["issues"].append(
                f"Column '{column_name}' has inconsistent categories: {', '.join(inconsistencies[:3])}"
            )
            profile["recommendations"].append(
                f"Standardize categories in '{column_name}' (case, whitespace, spelling)"
            )

    elif pd.api.types.is_datetime64_any_dtype(series):
        profile["statistics"] = _datetime_statistics(series)

        # Check for future dates if reasonable
        if "max_date" in profile["statistics"]:
            max_date = pd.to_datetime(profile["statistics"]["max_date"])
            if max_date > pd.Timestamp.now():
                profile["has_issues"] = True
                profile["issues"].append(f"Column '{column_name}' contains future dates")
                profile["recommendations"].append(
                    f"Verify future dates in '{column_name}' are intentional"
                )

    # High missing percentage warning
    if profile["null_percentage"] > 50:
        profile["has_issues"] = True
        profile["issues"].append(f"Column '{column_name}' is {profile['null_percentage']:.1f}% missing")
        profile["recommendations"].append(
            f"Column '{column_name}' has high missing rate - consider imputation or removal"
        )

    return profile


def _numeric_statistics(series: pd.Series) -> dict[str, Any]:
    """Calculate statistics for numeric columns."""
    clean_series = pd.to_numeric(series, errors='coerce').dropna()

    if clean_series.empty:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "skewness": None,
            "kurtosis": None,
        }

    return {
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "std": float(clean_series.std()),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "q25": float(clean_series.quantile(0.25)),
        "q75": float(clean_series.quantile(0.75)),
        "skewness": float(clean_series.skew()) if len(clean_series) > 2 else None,
        "kurtosis": float(clean_series.kurtosis()) if len(clean_series) > 3 else None,
    }


def _categorical_statistics(series: pd.Series) -> dict[str, Any]:
    """Calculate statistics for categorical columns."""
    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "unique_values": 0,
            "top_values": [],
            "mode": None,
            "mode_frequency": 0,
        }

    value_counts = clean_series.value_counts()
    top_values = [
        {"value": str(val), "count": int(count), "percentage": float(count / len(clean_series) * 100)}
        for val, count in value_counts.head(10).items()
    ]

    return {
        "unique_values": int(clean_series.nunique()),
        "top_values": top_values,
        "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
        "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
    }


def _datetime_statistics(series: pd.Series) -> dict[str, Any]:
    """Calculate statistics for datetime columns."""
    clean_series = pd.to_datetime(series, errors='coerce').dropna()

    if clean_series.empty:
        return {
            "min_date": None,
            "max_date": None,
            "date_range_days": None,
        }

    return {
        "min_date": str(clean_series.min()),
        "max_date": str(clean_series.max()),
        "date_range_days": int((clean_series.max() - clean_series.min()).days),
    }


def _detect_numeric_outliers(series: pd.Series) -> dict[str, Any]:
    """
    Detect outliers using IQR method.
    """
    clean_series = pd.to_numeric(series, errors='coerce').dropna()

    if clean_series.empty or len(clean_series) < 4:
        return {
            "method": "IQR",
            "count": 0,
            "percentage": 0.0,
            "outlier_values": [],
        }

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]

    return {
        "method": "IQR (1.5 × IQR rule)",
        "count": int(len(outliers)),
        "percentage": float(len(outliers) / len(clean_series) * 100),
        "outlier_values": [float(v) for v in outliers.head(10).tolist()],
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def _detect_category_inconsistencies(series: pd.Series) -> list[str]:
    """
    Detect likely inconsistent categories (case, whitespace issues).
    """
    clean_series = series.dropna().astype(str)

    if clean_series.empty:
        return []

    # Normalize: strip whitespace and lowercase
    normalized = clean_series.str.strip().str.lower()
    original_values = clean_series.unique()
    normalized_values = normalized.unique()

    # If normalized has fewer unique values, we have inconsistencies
    inconsistencies = []
    if len(normalized_values) < len(original_values):
        # Group by normalized version
        for norm_val in normalized_values:
            matching = clean_series[normalized == norm_val].unique()
            if len(matching) > 1:
                inconsistencies.append(f"[{', '.join(matching[:3])}]")

    return inconsistencies[:5]  # Return up to 5 examples


def _analyze_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze missing value patterns across the dataset.
    """
    total_cells = df.size
    total_missing = df.isna().sum().sum()

    missing_by_column = df.isna().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0].to_dict()

    # Calculate completeness by row
    rows_with_missing = df.isna().any(axis=1).sum()

    return {
        "total_cells": int(total_cells),
        "total_missing": int(total_missing),
        "missing_percentage": float(total_missing / total_cells * 100) if total_cells > 0 else 0.0,
        "columns_with_missing": {str(k): int(v) for k, v in columns_with_missing.items()},
        "columns_with_missing_count": len(columns_with_missing),
        "rows_with_missing": int(rows_with_missing),
        "complete_rows": int(len(df) - rows_with_missing),
        "complete_rows_percentage": float((len(df) - rows_with_missing) / len(df) * 100) if len(df) > 0 else 0.0,
    }


def _analyze_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze duplicate rows in the dataset.
    """
    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())

    # Check for duplicate user_id if column exists
    user_id_duplicates = 0
    if "user_id" in df.columns:
        user_id_duplicates = int(df["user_id"].duplicated().sum())

    # Check for duplicate session_id if column exists
    session_id_duplicates = 0
    if "session_id" in df.columns:
        session_id_duplicates = int(df["session_id"].duplicated().sum())

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": float(duplicate_count / len(df) * 100) if len(df) > 0 else 0.0,
        "unique_rows": int(len(df) - duplicate_count),
        "user_id_duplicates": user_id_duplicates,
        "session_id_duplicates": session_id_duplicates,
    }


def _calculate_quality_score(
    df: pd.DataFrame,
    column_profiles: list[dict],
    missing_analysis: dict,
    duplicate_analysis: dict,
) -> float:
    """
    Calculate overall data quality score (0-100).

    Scoring components:
    - Completeness (30%): (100 - missing_percentage)
    - Uniqueness (20%): (100 - duplicate_percentage)
    - Validity (30%): (columns without issues / total columns) * 100
    - Consistency (20%): No constant columns, no severe outliers
    """
    if df.empty or not column_profiles:
        return 0.0

    # Completeness score (30%)
    completeness = 100 - missing_analysis.get("missing_percentage", 100)
    completeness_score = (completeness / 100) * 30

    # Uniqueness score (20%)
    uniqueness = 100 - duplicate_analysis.get("duplicate_percentage", 100)
    uniqueness_score = (uniqueness / 100) * 20

    # Validity score (30%): columns without issues
    columns_without_issues = sum(1 for col in column_profiles if not col["has_issues"])
    validity = (columns_without_issues / len(column_profiles)) * 100 if column_profiles else 0
    validity_score = (validity / 100) * 30

    # Consistency score (20%): penalize for severe data quality issues
    consistency = 100.0

    # Penalize for constant columns
    constant_columns = sum(1 for col in column_profiles if col["unique_count"] == 1)
    if constant_columns > 0:
        consistency -= min(constant_columns * 10, 50)  # Max 50% penalty

    # Penalize for high missing rate columns
    high_missing_cols = sum(1 for col in column_profiles if col["null_percentage"] > 50)
    if high_missing_cols > 0:
        consistency -= min(high_missing_cols * 5, 30)  # Max 30% penalty

    consistency_score = max(consistency / 100, 0) * 20

    # Total score
    total_score = completeness_score + uniqueness_score + validity_score + consistency_score

    return round(total_score, 2)


def get_quality_grade(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 90:
        return "A (Excellent)"
    elif score >= 80:
        return "B (Good)"
    elif score >= 70:
        return "C (Fair)"
    elif score >= 60:
        return "D (Poor)"
    else:
        return "F (Critical Issues)"


def suggest_cleaning_actions(profile: dict) -> list[dict[str, str]]:
    """
    Generate actionable cleaning suggestions based on profile.

    Returns:
        List of dictionaries with 'action', 'reason', and 'impact' keys
    """
    suggestions = []

    # Missing value suggestions
    missing_pct = profile["missing_analysis"].get("missing_percentage", 0)
    if missing_pct > 10:
        suggestions.append({
            "action": "Handle missing values",
            "reason": f"{missing_pct:.1f}% of data is missing",
            "impact": "Improves data completeness and statistical power",
        })

    # Duplicate suggestions
    dup_pct = profile["duplicate_analysis"].get("duplicate_percentage", 0)
    if dup_pct > 0:
        suggestions.append({
            "action": "Remove or investigate duplicate rows",
            "reason": f"{dup_pct:.1f}% of rows are exact duplicates",
            "impact": "Ensures independent observations for statistical tests",
        })

    # User ID duplicates
    user_id_dup = profile["duplicate_analysis"].get("user_id_duplicates", 0)
    if user_id_dup > 0:
        suggestions.append({
            "action": "Investigate repeated user_id values",
            "reason": f"{user_id_dup:,} duplicate user_id value(s) indicate repeated observations",
            "impact": "Critical for independence assumption in hypothesis testing",
        })

    # Constant columns
    constant_cols = [
        col["column_name"] for col in profile["column_profiles"]
        if col["unique_count"] == 1
    ]
    if constant_cols:
        suggestions.append({
            "action": f"Consider removing constant columns: {', '.join(constant_cols)}",
            "reason": "Constant columns provide no information",
            "impact": "Simplifies analysis and reduces data size",
        })

    # High missing rate columns
    high_missing_cols = [
        col["column_name"] for col in profile["column_profiles"]
        if col["null_percentage"] > 50
    ]
    if high_missing_cols:
        suggestions.append({
            "action": f"Address high missing rates in: {', '.join(high_missing_cols[:3])}",
            "reason": "These columns are >50% missing",
            "impact": "Improves reliability of analysis involving these fields",
        })

    # Category inconsistencies
    inconsistent_cols = [
        col["column_name"] for col in profile["column_profiles"]
        if "inconsistent categories" in str(col.get("issues", []))
    ]
    if inconsistent_cols:
        suggestions.append({
            "action": f"Standardize categories in: {', '.join(inconsistent_cols[:3])}",
            "reason": "Inconsistent formatting detected (case, whitespace)",
            "impact": "Ensures accurate grouping and segmentation",
        })

    return suggestions


def calculate_data_reliability_score(profile: dict) -> dict[str, Any]:
    """
    Calculate specific reliability metrics for statistical analysis.

    Returns metrics for:
    - Independence reliability
    - Completeness reliability
    - Consistency reliability
    """
    reliability = {
        "independence_score": 100.0,
        "independence_warnings": [],
        "completeness_score": 100.0,
        "completeness_warnings": [],
        "consistency_score": 100.0,
        "consistency_warnings": [],
        "overall_reliability": "High",
    }

    # Independence checks
    dup_analysis = profile.get("duplicate_analysis", {})
    if dup_analysis.get("user_id_duplicates", 0) > 0:
        reliability["independence_score"] -= 30
        reliability["independence_warnings"].append(
            "Repeated user_id values detected - independence assumption may be violated"
        )

    if dup_analysis.get("duplicate_rows", 0) > 0:
        reliability["independence_score"] -= 20
        reliability["independence_warnings"].append(
            "Exact duplicate rows detected - may represent repeated observations"
        )

    # Completeness checks
    missing_analysis = profile.get("missing_analysis", {})
    missing_pct = missing_analysis.get("missing_percentage", 0)
    if missing_pct > 20:
        reliability["completeness_score"] -= 40
        reliability["completeness_warnings"].append(
            f"{missing_pct:.1f}% of data is missing - may affect statistical power"
        )
    elif missing_pct > 10:
        reliability["completeness_score"] -= 20
        reliability["completeness_warnings"].append(
            f"{missing_pct:.1f}% of data is missing"
        )

    # Consistency checks
    issues_count = len(profile.get("issues", []))
    if issues_count > 5:
        reliability["consistency_score"] -= 30
        reliability["consistency_warnings"].append(
            f"{issues_count} data quality issues detected across columns"
        )
    elif issues_count > 0:
        reliability["consistency_score"] -= 15
        reliability["consistency_warnings"].append(
            f"{issues_count} data quality issues detected"
        )

    # Overall reliability
    avg_score = (
        reliability["independence_score"] +
        reliability["completeness_score"] +
        reliability["consistency_score"]
    ) / 3

    if avg_score >= 80:
        reliability["overall_reliability"] = "High"
    elif avg_score >= 60:
        reliability["overall_reliability"] = "Medium"
    else:
        reliability["overall_reliability"] = "Low"

    return reliability
