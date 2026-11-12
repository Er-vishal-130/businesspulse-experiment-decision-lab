"""
Unit Tests for Data Quality Module
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_quality import (
    profile_dataframe,
    calculate_data_reliability_score,
    get_quality_grade,
    suggest_cleaning_actions,
)


class TestDataQualityProfiling:
    """Test data quality profiling functionality."""

    def test_profile_empty_dataframe(self):
        """Test profiling with empty dataframe."""
        df = pd.DataFrame()
        profile = profile_dataframe(df)

        assert profile["total_rows"] == 0
        assert profile["total_columns"] == 0
        assert len(profile["issues"]) > 0

    def test_profile_basic_dataframe(self):
        """Test profiling with basic valid data."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "converted": [1, 0, 1, 0, 1],
            "revenue": [100.0, 0.0, 150.0, 0.0, 200.0],
        })

        profile = profile_dataframe(df)

        assert profile["total_rows"] == 5
        assert profile["total_columns"] == 3
        assert profile["overall_quality_score"] > 0
        assert len(profile["column_profiles"]) == 3

    def test_profile_with_missing_values(self):
        """Test profiling detects missing values."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, None, 5],
            "converted": [1, 0, None, 0, 1],
        })

        profile = profile_dataframe(df)
        missing_analysis = profile["missing_analysis"]

        assert missing_analysis["total_missing"] == 2
        assert missing_analysis["columns_with_missing_count"] == 2

    def test_profile_with_duplicates(self):
        """Test profiling detects duplicate rows."""
        df = pd.DataFrame({
            "user_id": [1, 2, 2, 3],
            "converted": [1, 0, 0, 1],
        })

        profile = profile_dataframe(df)
        dup_analysis = profile["duplicate_analysis"]

        assert dup_analysis["duplicate_rows"] == 1
        assert dup_analysis["user_id_duplicates"] == 1

    def test_outlier_detection(self):
        """Test numeric outlier detection."""
        df = pd.DataFrame({
            "revenue": [100, 105, 98, 102, 1000],  # 1000 is outlier
        })

        profile = profile_dataframe(df)
        revenue_profile = next(p for p in profile["column_profiles"] if p["column_name"] == "revenue")

        assert revenue_profile["outliers"] is not None
        assert revenue_profile["outliers"]["count"] > 0

    def test_constant_column_detection(self):
        """Test detection of constant-value columns."""
        df = pd.DataFrame({
            "constant": [1, 1, 1, 1],
            "variable": [1, 2, 3, 4],
        })

        profile = profile_dataframe(df)
        constant_col = next(p for p in profile["column_profiles"] if p["column_name"] == "constant")

        assert constant_col["unique_count"] == 1
        assert constant_col["has_issues"] is True

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # Perfect data
        perfect_df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "converted": [1, 0, 1, 0, 1],
        })

        profile = profile_dataframe(perfect_df)
        assert profile["overall_quality_score"] >= 90

        # Poor quality data
        poor_df = pd.DataFrame({
            "user_id": [1, 1, None, None, None],
            "converted": [1, 1, None, None, None],
        })

        profile = profile_dataframe(poor_df)
        assert profile["overall_quality_score"] < 70

    def test_quality_grade_assignment(self):
        """Test quality grade conversion."""
        assert get_quality_grade(95) == "A (Excellent)"
        assert get_quality_grade(85) == "B (Good)"
        assert get_quality_grade(75) == "C (Fair)"
        assert get_quality_grade(65) == "D (Poor)"
        assert get_quality_grade(45) == "F (Critical Issues)"

    def test_cleaning_suggestions(self):
        """Test cleaning action suggestions."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 3],  # Duplicates
            "converted": [1, None, 0, 1],  # Missing values
        })

        profile = profile_dataframe(df)
        suggestions = suggest_cleaning_actions(profile)

        assert len(suggestions) > 0
        assert any("missing" in s["reason"].lower() for s in suggestions)
        assert any("duplicate" in s["reason"].lower() for s in suggestions)

    def test_reliability_score(self):
        """Test data reliability scoring."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 3],
            "converted": [1, 0, 1, 0],
        })

        profile = profile_dataframe(df)
        reliability = calculate_data_reliability_score(profile)

        assert "independence_score" in reliability
        assert "completeness_score" in reliability
        assert "consistency_score" in reliability
        assert reliability["overall_reliability"] in ["High", "Medium", "Low"]


class TestCategoryInconsistency:
    """Test category inconsistency detection."""

    def test_case_inconsistency(self):
        """Test detection of case inconsistencies."""
        df = pd.DataFrame({
            "device": ["Mobile", "mobile", "MOBILE", "Desktop"],
        })

        profile = profile_dataframe(df)
        device_profile = next(p for p in profile["column_profiles"] if p["column_name"] == "device")

        assert device_profile["has_issues"] is True
        assert any("inconsistent categories" in issue.lower() for issue in device_profile["issues"])

    def test_whitespace_inconsistency(self):
        """Test detection of whitespace issues."""
        df = pd.DataFrame({
            "source": ["Organic", " Organic", "Organic ", "Paid"],
        })

        profile = profile_dataframe(df)
        source_profile = next(p for p in profile["column_profiles"] if p["column_name"] == "source")

        # Should detect that "Organic" appears multiple times with different whitespace
        assert source_profile["unique_count"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
