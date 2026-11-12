"""
Unit Tests for KPI Intelligence Module
"""
import pytest
import pandas as pd
import numpy as np
from modules.kpi_intelligence import (
    detect_available_kpis,
    calculate_kpi_values,
    calculate_kpi_trends,
    detect_kpi_anomalies,
    calculate_kpi_health_score,
)


class TestKPIDetection:
    """Test KPI availability detection."""

    def test_detect_basic_kpis(self):
        """Test detection of basic conversion KPIs."""
        df = pd.DataFrame({
            "converted": [1, 0, 1, 0, 1],
            "revenue": [100, 0, 150, 0, 200],
        })

        kpis = detect_available_kpis(df)

        assert kpis["Conversion Rate"]["available"] is True
        assert kpis["Total Revenue"]["available"] is True
        assert kpis["Total Users"]["available"] is False  # No user_id

    def test_detect_all_kpis(self):
        """Test detection with all KPI columns present."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "session_id": [101, 102, 103],
            "converted": [1, 0, 1],
            "revenue": [100, 0, 150],
            "bounce": [0, 1, 0],
            "session_duration": [120, 30, 180],
            "pages_viewed": [5, 1, 8],
            "error_flag": [0, 1, 0],
        })

        kpis = detect_available_kpis(df)

        available_count = sum(1 for kpi in kpis.values() if kpi["available"])
        assert available_count >= 8

    def test_missing_columns_reported(self):
        """Test that missing columns are properly reported."""
        df = pd.DataFrame({"converted": [1, 0, 1]})

        kpis = detect_available_kpis(df)
        aov_kpi = kpis["Average Order Value"]

        assert aov_kpi["available"] is False
        assert "revenue" in aov_kpi["missing_columns"]


class TestKPICalculation:
    """Test KPI value calculations."""

    def test_conversion_rate_calculation(self):
        """Test conversion rate calculation."""
        df = pd.DataFrame({
            "converted": [1, 0, 1, 0, 1, 1, 0, 0],  # 4/8 = 50%
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        conv_rate = kpi_values["Conversion Rate"]["value"]
        assert conv_rate == 0.5

    def test_revenue_calculation(self):
        """Test total revenue calculation."""
        df = pd.DataFrame({
            "revenue": [100, 200, 300],
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        total_revenue = kpi_values["Total Revenue"]["value"]
        assert total_revenue == 600

    def test_aov_calculation(self):
        """Test Average Order Value calculation."""
        df = pd.DataFrame({
            "converted": [1, 0, 1, 0, 1],
            "revenue": [100, 0, 200, 0, 300],  # 3 conversions, 600 revenue
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        aov = kpi_values["Average Order Value"]["value"]
        assert aov == 200  # 600/3

    def test_bounce_rate_calculation(self):
        """Test bounce rate calculation."""
        df = pd.DataFrame({
            "bounce": [1, 0, 1, 0],  # 2/4 = 50%
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        bounce_rate = kpi_values["Bounce Rate"]["value"]
        assert bounce_rate == 0.5

    def test_formatted_values(self):
        """Test KPI value formatting."""
        df = pd.DataFrame({
            "converted": [1, 0, 1, 0],  # 50%
            "revenue": [1000, 0, 2000, 0],
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        assert "50.00%" in kpi_values["Conversion Rate"]["formatted_value"]
        assert "$3,000.00" in kpi_values["Total Revenue"]["formatted_value"]


class TestKPITrends:
    """Test KPI trend analysis."""

    def test_trend_calculation(self):
        """Test time-series trend calculation."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "converted": [1, 0, 1, 1, 0],
            "revenue": [100, 0, 200, 150, 0],
        })

        kpi_availability = detect_available_kpis(df)
        trends = calculate_kpi_trends(df, kpi_availability, date_column="date", frequency="D")

        assert "Conversion Rate" in trends
        assert len(trends["Conversion Rate"]) == 5

    def test_trend_without_date_column(self):
        """Test trend calculation handles missing date column."""
        df = pd.DataFrame({
            "converted": [1, 0, 1],
        })

        kpi_availability = detect_available_kpis(df)
        trends = calculate_kpi_trends(df, kpi_availability)

        assert len(trends) == 0


class TestKPIAnomalyDetection:
    """Test KPI anomaly detection."""

    def test_zscore_anomaly_detection(self):
        """Test z-score based anomaly detection in KPI trends."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        # Normal rates around 50%, one outlier at 95%
        rates = [0.5, 0.48, 0.52, 0.49, 0.95, 0.51, 0.50, 0.48, 0.52, 0.49]

        df = pd.DataFrame({
            "date": dates,
            "converted": [int(r * 100) for r in rates],
            "total": [100] * 10,
        })

        df["date_period"] = dates
        trends = {"Test KPI": df[["date_period"]].assign(value=rates)}

        anomalies = detect_kpi_anomalies(trends, method="zscore", threshold=2.0)

        assert "Test KPI" in anomalies
        assert len(anomalies["Test KPI"]) > 0


class TestKPIHealthScore:
    """Test KPI health scoring."""

    def test_health_score_all_available(self):
        """Test health score when all KPIs are available."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "converted": [1, 0, 1],
            "revenue": [100, 0, 200],
            "bounce": [0, 1, 0],
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)
        health = calculate_kpi_health_score(kpi_values)

        assert health["overall_score"] > 0
        assert health["health_grade"] in ["Excellent", "Good", "Fair", "Poor"]

    def test_health_score_with_targets(self):
        """Test health score with target achievement."""
        df = pd.DataFrame({
            "converted": [1, 0, 1, 0],
            "revenue": [100, 0, 200, 0],
        })

        kpi_availability = detect_available_kpis(df)
        kpi_values = calculate_kpi_values(df, kpi_availability)

        targets = {
            "Conversion Rate": 0.4,  # Target 40%, actual 50%
            "Total Revenue": 200,  # Target 200, actual 300
        }

        health = calculate_kpi_health_score(kpi_values, targets=targets)

        assert health["target_achievement_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
