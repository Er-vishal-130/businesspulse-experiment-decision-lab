"""
Unit Tests for Enhanced Investigation Module
"""
import pytest
import pandas as pd
from modules.investigation import (
    enhanced_root_cause_investigation,
    drill_down_segment,
)


class TestRootCauseInvestigation:
    """Test enhanced root cause investigation."""

    def test_basic_investigation(self):
        """Test basic root cause investigation."""
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "device": ["Mobile"] * 50 + ["Desktop"] * 50 + ["Mobile"] * 50 + ["Desktop"] * 50,
            "converted": [1] * 5 + [0] * 45 + [1] * 5 + [0] * 45 + [1] * 40 + [0] * 10 + [1] * 5 + [0] * 45,
        })

        result = enhanced_root_cause_investigation(
            df, metric_col="converted", dimensions=["device"]
        )

        assert result["status"] == "success"
        assert len(result["dimension_rankings"]) > 0
        assert len(result["top_segments"]) > 0

    def test_multiple_dimensions(self):
        """Test investigation across multiple dimensions."""
        df = pd.DataFrame({
            "group": ["Control"] * 200 + ["Treatment"] * 200,
            "device": (["Mobile"] * 100 + ["Desktop"] * 100) * 2,
            "country": (["USA"] * 50 + ["UK"] * 50) * 4,
            "converted": [1] * 20 + [0] * 180 + [1] * 80 + [0] * 120,
        })

        result = enhanced_root_cause_investigation(
            df, dimensions=["device", "country"]
        )

        assert len(result["dimension_rankings"]) == 2
        assert result["dimensions_analyzed"] == 2

    def test_investigation_timeline(self):
        """Test investigation timeline generation."""
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "device": ["Mobile"] * 100 + ["Desktop"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 80 + [0] * 20,
        })

        result = enhanced_root_cause_investigation(df, dimensions=["device"])

        assert "investigation_timeline" in result
        assert len(result["investigation_timeline"]) > 0
        assert result["investigation_timeline"][0]["step"] == 1

    def test_next_steps_generation(self):
        """Test next investigation steps generation."""
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "device": ["Mobile"] * 100 + ["Desktop"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 60 + [0] * 40,
        })

        result = enhanced_root_cause_investigation(df, dimensions=["device"])

        assert "next_steps" in result
        assert len(result["next_steps"]) > 0
        assert "priority" in result["next_steps"][0]

    def test_contribution_ranking(self):
        """Test contribution-based ranking."""
        df = pd.DataFrame({
            "group": ["Control"] * 150 + ["Treatment"] * 150,
            "device": (["Mobile"] * 50 + ["Desktop"] * 50 + ["Tablet"] * 50) * 2,
            "converted": [1] * 5 + [0] * 45 + [1] * 5 + [0] * 45 + [1] * 5 + [0] * 45 +
                        [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40 + [1] * 5 + [0] * 45,
        })

        result = enhanced_root_cause_investigation(df, dimensions=["device"])

        top_segment = result["top_segments"][0]
        assert "contribution" in top_segment
        assert "segment" in top_segment

    def test_evidence_summary(self):
        """Test evidence summary generation."""
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "device": ["Mobile"] * 100 + ["Desktop"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 70 + [0] * 30,
        })

        result = enhanced_root_cause_investigation(df, dimensions=["device"])

        assert "evidence_summary" in result
        evidence = result["evidence_summary"]
        assert "dimensions_analyzed" in evidence
        assert "total_contribution" in evidence
        assert "evidence_strength" in evidence


class TestDrillDown:
    """Test segment drill-down functionality."""

    def test_drill_down_analysis(self):
        """Test drilling down into specific segment."""
        df = pd.DataFrame({
            "device": ["Mobile"] * 400 + ["Desktop"] * 400,
            "traffic_source": (["Organic"] * 100 + ["Paid"] * 100) * 4,
            "country": (["USA"] * 50 + ["UK"] * 50) * 8,
            "converted": [1] * 20 + [0] * 180 + [1] * 15 + [0] * 185 +
                        [1] * 10 + [0] * 190 + [1] * 12 + [0] * 188,
        })

        result = drill_down_segment(
            df,
            primary_dimension="device",
            primary_segment="Mobile",
            secondary_dimensions=["traffic_source", "country"],
        )

        assert result["status"] == "success"
        assert result["primary_segment"] == "Mobile"
        assert "sub_analysis" in result

    def test_drill_down_invalid_segment(self):
        """Test drill-down with invalid segment."""
        df = pd.DataFrame({
            "device": ["Mobile", "Desktop"],
            "converted": [1, 0],
        })

        result = drill_down_segment(
            df,
            primary_dimension="device",
            primary_segment="Tablet",
            secondary_dimensions=["traffic_source"],
        )

        assert result["status"] == "error"


class TestContributionAnalysis:
    """Test contribution analysis accuracy."""

    def test_contribution_calculation(self):
        """Test accurate contribution calculation."""
        # Control: 10% conversion, Treatment: 20% conversion on Mobile (100 users)
        # Expected contribution: 100 * (0.20 - 0.10) = +10 conversions
        df = pd.DataFrame({
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "device": ["Mobile"] * 100 + ["Mobile"] * 100,
            "converted": [1] * 10 + [0] * 90 + [1] * 20 + [0] * 80,
        })

        result = enhanced_root_cause_investigation(df, dimensions=["device"])

        mobile_segment = result["top_segments"][0]
        assert abs(mobile_segment["contribution"] - 10) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
