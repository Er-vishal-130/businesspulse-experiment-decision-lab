"""
Unit Tests for Funnel and Cohort Analytics
"""
import pytest
import pandas as pd
from modules.funnel_analytics import (
    analyze_conversion_funnel,
    identify_funnel_bottlenecks,
    compare_funnels,
)
from modules.cohort_analytics import (
    create_cohort_analysis,
    analyze_repeat_behavior,
)


class TestFunnelAnalytics:
    """Test funnel analytics functionality."""

    def test_basic_funnel_analysis(self):
        """Test basic funnel analysis."""
        df = pd.DataFrame({
            "viewed": [1, 1, 1, 1, 1] * 20,
            "clicked": [1, 1, 1, 0, 0] * 20,
            "converted": [1, 1, 0, 0, 0] * 20,
        })

        result = analyze_conversion_funnel(df, funnel_steps=["viewed", "clicked", "converted"])

        assert result["status"] == "success"
        metrics = result["overall_metrics"]["step_metrics"]
        assert len(metrics) == 3
        assert metrics[0]["users"] == 100  # All viewed
        assert metrics[1]["users"] == 60   # 60% clicked
        assert metrics[2]["users"] == 40   # 40% converted

    def test_funnel_with_groups(self):
        """Test funnel analysis with group comparison."""
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "viewed": [1] * 200,
            "clicked": [1] * 40 + [0] * 60 + [1] * 60 + [0] * 40,
            "converted": [1] * 20 + [0] * 80 + [1] * 40 + [0] * 60,
        })

        result = analyze_conversion_funnel(
            df,
            funnel_steps=["viewed", "clicked", "converted"],
            group_by="group"
        )

        assert "segment_metrics" in result
        assert "Control" in result["segment_metrics"]
        assert "Treatment" in result["segment_metrics"]

    def test_bottleneck_identification(self):
        """Test funnel bottleneck identification."""
        # Create funnel with clear bottleneck at "clicked" stage
        funnel_metrics = {
            "step_metrics": [
                {"step_name": "viewed", "drop_off_from_previous": 0.0, "users_dropped": 0},
                {"step_name": "clicked", "drop_off_from_previous": 0.6, "users_dropped": 60},
                {"step_name": "converted", "drop_off_from_previous": 0.2, "users_dropped": 8},
            ]
        }

        bottlenecks = identify_funnel_bottlenecks(funnel_metrics, threshold=0.3)

        assert len(bottlenecks) == 1
        assert bottlenecks[0]["step_name"] == "clicked"
        assert bottlenecks[0]["severity"] in ["High", "Critical"]

    def test_funnel_comparison(self):
        """Test Control vs Treatment funnel comparison."""
        control_funnel = {
            "step_metrics": [
                {"step_name": "viewed", "conversion_from_previous": 1.0, "users": 100},
                {"step_name": "clicked", "conversion_from_previous": 0.5, "users": 50},
                {"step_name": "converted", "conversion_from_previous": 0.4, "users": 20},
            ],
            "overall_conversion": 0.2,
        }

        treatment_funnel = {
            "step_metrics": [
                {"step_name": "viewed", "conversion_from_previous": 1.0, "users": 100},
                {"step_name": "clicked", "conversion_from_previous": 0.7, "users": 70},
                {"step_name": "converted", "conversion_from_previous": 0.5, "users": 35},
            ],
            "overall_conversion": 0.35,
        }

        comparison = compare_funnels(control_funnel, treatment_funnel)

        assert comparison["status"] == "success"
        assert len(comparison["step_comparisons"]) == 3
        assert comparison["biggest_change"] is not None


class TestCohortAnalytics:
    """Test cohort analytics functionality."""

    def test_cohort_creation(self):
        """Test cohort creation and retention tracking."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")

        # Create user activity data
        data = []
        for user_id in range(1, 51):
            first_date = dates[user_id % 30]  # Users join in first 30 days
            active_dates = [first_date + pd.Timedelta(days=d) for d in [0, 7, 14, 30] if first_date + pd.Timedelta(days=d) <= dates[-1]]
            for date in active_dates:
                data.append({"user_id": user_id, "date": date})

        df = pd.DataFrame(data)

        result = create_cohort_analysis(df, cohort_period="M", retention_periods=3)

        assert result["status"] == "success"
        assert "cohort_table" in result
        assert len(result["cohort_sizes"]) > 0

    def test_repeat_behavior_analysis(self):
        """Test repeat purchase behavior analysis."""
        df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 3, 4, 5, 5],
            "converted": [1] * 9,
            "date": pd.date_range("2024-01-01", periods=9, freq="D"),
        })

        result = analyze_repeat_behavior(df)

        assert result["status"] == "success"
        assert result["total_converting_users"] == 5
        assert result["repeat_users"] > 0
        assert result["repeat_rate"] > 0

    def test_one_time_vs_repeat_users(self):
        """Test classification of one-time vs repeat users."""
        df = pd.DataFrame({
            "user_id": [1, 2, 2, 3, 3, 3, 4, 5],
            "converted": [1] * 8,
        })

        result = analyze_repeat_behavior(df)

        assert result["one_time_users"] == 3  # Users 1, 4, 5
        assert result["repeat_users"] == 2    # Users 2, 3


class TestFunnelAutoDetection:
    """Test automatic funnel step detection."""

    def test_auto_detect_binary_columns(self):
        """Test automatic detection of binary funnel steps."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "visited": [1, 1, 1],
            "clicked": [1, 0, 1],
            "converted": [1, 0, 0],
            "device": ["Mobile", "Desktop", "Mobile"],  # Not binary
        })

        result = analyze_conversion_funnel(df)  # No explicit steps

        assert result["status"] == "success"
        assert len(result["funnel_steps"]) >= 2  # Should detect visited, clicked, converted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
