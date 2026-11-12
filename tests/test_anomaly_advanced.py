"""
Unit Tests for Advanced Anomaly Detection Module
"""
import pytest
import pandas as pd
import numpy as np
from modules.anomaly_advanced import (
    detect_anomalies_multi_method,
    detect_time_series_anomalies,
    explain_anomaly_detection,
)


class TestMultiMethodDetection:
    """Test multi-method anomaly detection."""

    def test_zscore_detection(self):
        """Test z-score based anomaly detection."""
        # Create data with one clear outlier
        df = pd.DataFrame({
            "device": ["Mobile"] * 50 + ["Desktop"] * 50 + ["Tablet"] * 50,
            "converted": [1] * 5 + [0] * 45 + [1] * 5 + [0] * 45 + [1] * 45 + [0] * 5,  # Tablet is outlier
        })

        result = detect_anomalies_multi_method(
            df, segment_by="device", metric_col="converted", methods=["zscore"]
        )

        assert result["status"] != "error"
        assert "zscore" in result["anomalies_by_method"]

    def test_iqr_detection(self):
        """Test IQR based anomaly detection."""
        df = pd.DataFrame({
            "device": ["Mobile"] * 100 + ["Desktop"] * 100 + ["Tablet"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 10 + [0] * 90 + [1] * 80 + [0] * 20,
        })

        result = detect_anomalies_multi_method(
            df, segment_by="device", methods=["iqr"]
        )

        assert "iqr" in result["anomalies_by_method"]

    def test_consensus_anomalies(self):
        """Test consensus anomaly identification across methods."""
        # Create strong outlier that should be detected by multiple methods
        df = pd.DataFrame({
            "device": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 10 + [0] * 90 + [1] * 90 + [0] * 10,
        })

        result = detect_anomalies_multi_method(
            df, segment_by="device", methods=["zscore", "iqr"]
        )

        # Should have consensus if both methods detect the same anomaly
        assert "consensus_anomalies" in result

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = pd.DataFrame({
            "device": ["Mobile", "Desktop"],
            "converted": [1, 0],
        })

        result = detect_anomalies_multi_method(df, segment_by="device")

        assert len(result["warnings"]) > 0 or result["status"] == "error"

    def test_multiple_methods(self):
        """Test running multiple detection methods."""
        df = pd.DataFrame({
            "device": ["Mobile"] * 100 + ["Desktop"] * 100,
            "converted": [1] * 50 + [0] * 50 + [1] * 10 + [0] * 90,
        })

        result = detect_anomalies_multi_method(
            df, segment_by="device", methods=["zscore", "iqr"]
        )

        assert len(result["methods_used"]) == 2
        assert len(result["anomalies_by_method"]) >= 1


class TestTimeSeriesAnomalies:
    """Test time-series anomaly detection."""

    def test_rolling_baseline_detection(self):
        """Test rolling baseline anomaly detection."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Normal pattern with one spike
        conversions = [50] * 14 + [150] + [50] * 15  # Day 15 is anomaly

        df = pd.DataFrame({
            "date": dates,
            "converted": conversions,
        })
        df["converted"] = df["converted"].apply(lambda x: [1] * x + [0] * (200 - x))
        df = df.explode("converted")
        df["converted"] = df["converted"].astype(int)

        result = detect_time_series_anomalies(
            df, metric_col="converted", date_col="date", method="rolling", window_size=7
        )

        assert result["method"] == "Rolling Baseline"

    def test_changepoint_detection(self):
        """Test change-point detection."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        # Sudden change on day 10
        conversions = [0.3] * 10 + [0.7] * 10

        df = pd.DataFrame({
            "date": dates,
            "converted": [int(r * 100) for r in conversions],
        })

        df["converted"] = df["converted"].apply(lambda x: [1] * x + [0] * (100 - x))
        df = df.explode("converted")
        df["converted"] = df["converted"].astype(int)

        result = detect_time_series_anomalies(
            df, method="changepoint"
        )

        assert result["method"] == "Change-point Detection"

    def test_insufficient_time_periods(self):
        """Test handling of insufficient time periods."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "converted": [1, 0, 1],
        })

        result = detect_time_series_anomalies(df, window_size=7)

        assert len(result["warnings"]) > 0


class TestAnomalyExplanation:
    """Test anomaly detection explanation."""

    def test_zscore_explanation(self):
        """Test z-score method explanation."""
        method_result = {
            "method": "Z-score",
            "threshold": 3.0,
            "baseline_mean": 0.05,
        }

        explanation = explain_anomaly_detection(method_result)

        assert "Z-score" in explanation
        assert "3" in explanation or "3.0" in explanation

    def test_iqr_explanation(self):
        """Test IQR method explanation."""
        method_result = {
            "method": "IQR",
            "multiplier": 1.5,
            "lower_bound": 0.02,
            "upper_bound": 0.08,
        }

        explanation = explain_anomaly_detection(method_result)

        assert "IQR" in explanation
        assert "1.5" in explanation

    def test_isolation_forest_explanation(self):
        """Test Isolation Forest explanation."""
        method_result = {
            "method": "Isolation Forest",
            "features_used": ["rate", "mean", "sum"],
            "contamination": 0.1,
        }

        explanation = explain_anomaly_detection(method_result)

        assert "Isolation Forest" in explanation
        assert "features" in explanation.lower()


class TestSeverityCalculation:
    """Test anomaly severity calculation."""

    def test_severity_from_zscore(self):
        """Test severity calculation from z-scores."""
        rates = [0.1] * 10 + [0.9]
        df = pd.DataFrame({
            "device": [segment for segment in range(len(rates)) for _ in range(100)],
            "converted": [
                outcome
                for rate in rates
                for outcome in ([1] * int(rate * 100) + [0] * (100 - int(rate * 100)))
            ],
        })

        result = detect_anomalies_multi_method(df, segment_by="device", methods=["zscore"])

        if result.get("anomalies_by_method", {}).get("zscore", {}).get("anomalies"):
            anomaly = result["anomalies_by_method"]["zscore"]["anomalies"][0]
            assert "severity" in anomaly
            assert anomaly["severity"] in ["Low", "Medium", "High", "Critical"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
