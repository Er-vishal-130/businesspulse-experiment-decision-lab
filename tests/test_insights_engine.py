"""
Unit Tests for Insights Engine Module
"""
import pytest
import pandas as pd
from modules.insights_engine import (
    generate_insights_from_analysis,
    prioritize_investigation_steps,
    generate_executive_summary,
    Insight,
)


class TestInsightGeneration:
    """Test automated insight generation."""

    def test_conversion_improvement_insight(self):
        """Test insight generation for conversion improvement."""
        summary = {
            "control_conversion": 0.05,
            "treatment_conversion": 0.08,
            "absolute_lift": 0.03,
            "relative_lift": 60.0,
            "total_users": 10000,
        }

        hypothesis = {
            "p_value": 0.001,
        }

        insights = generate_insights_from_analysis(summary, hypothesis)

        assert len(insights) > 0
        conversion_insights = [i for i in insights if i.category == "Conversion Performance"]
        assert len(conversion_insights) > 0
        assert conversion_insights[0].priority in ["High", "Medium"]

    def test_conversion_decline_insight(self):
        """Test insight generation for conversion decline."""
        summary = {
            "control_conversion": 0.08,
            "treatment_conversion": 0.05,
            "absolute_lift": -0.03,
            "relative_lift": -37.5,
            "total_users": 5000,
        }

        hypothesis = {
            "p_value": 0.01,
        }

        insights = generate_insights_from_analysis(summary, hypothesis)

        decline_insights = [i for i in insights if "decline" in i.finding.lower()]
        assert len(decline_insights) > 0
        assert decline_insights[0].priority in ["Critical", "High"]

    def test_inconclusive_result_insight(self):
        """Test insight generation for inconclusive results."""
        summary = {
            "control_conversion": 0.05,
            "treatment_conversion": 0.051,
            "absolute_lift": 0.001,
            "relative_lift": 2.0,
            "total_users": 500,
        }

        hypothesis = {
            "p_value": 0.45,
        }

        insights = generate_insights_from_analysis(summary, hypothesis)

        inconclusive = [i for i in insights if "inconclusive" in i.finding.lower()]
        assert len(inconclusive) > 0

    def test_kpi_unavailable_insight(self):
        """Test insight generation for unavailable KPIs."""
        kpi_values = {
            "Total Revenue": {
                "status": "unavailable",
                "reason": "Missing revenue column",
            },
            "Conversion Rate": {
                "status": "available",
                "value": 0.05,
            },
        }

        insights = generate_insights_from_analysis(
            summary={}, hypothesis={}, kpi_values=kpi_values
        )

        kpi_insights = [i for i in insights if i.category == "Data Quality"]
        assert len(kpi_insights) > 0

    def test_anomaly_insights(self):
        """Test insight generation from anomaly detection."""
        anomaly_results = {
            "consensus_anomalies": [
                {
                    "segment": "Mobile",
                    "observed_rate": 0.15,
                    "expected_rate": 0.05,
                    "deviation": 0.10,
                    "methods_agreeing": ["zscore", "iqr"],
                }
            ]
        }

        insights = generate_insights_from_analysis(
            summary={}, hypothesis={}, anomaly_results=anomaly_results
        )

        anomaly_insights = [i for i in insights if i.category == "Anomaly Detection"]
        assert len(anomaly_insights) > 0


class TestInsightPriority:
    """Test insight prioritization."""

    def test_priority_ordering(self):
        """Test that insights are ordered by priority."""
        insights = [
            Insight("Low priority", {}, "Low", 50, [], [], "Test"),
            Insight("Critical priority", {}, "Critical", 90, [], [], "Test"),
            Insight("Medium priority", {}, "Medium", 70, [], [], "Test"),
            Insight("High priority", {}, "High", 80, [], [], "Test"),
        ]

        # Should be sorted: Critical, High, Medium, Low
        sorted_insights = sorted(
            insights,
            key=lambda x: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}[x.priority]
        )

        assert sorted_insights[0].priority == "Critical"
        assert sorted_insights[1].priority == "High"
        assert sorted_insights[2].priority == "Medium"
        assert sorted_insights[3].priority == "Low"


class TestInvestigationSteps:
    """Test investigation step generation."""

    def test_prioritize_investigation_steps(self):
        """Test generation of prioritized investigation steps."""
        insights = [
            Insight(
                "Treatment shows decline",
                {"control_rate": 0.08, "treatment_rate": 0.05},
                "Critical",
                95,
                ["Investigate root cause"],
                ["Statistical only"],
                "Conversion Performance",
            ),
            Insight(
                "Mobile anomaly detected",
                {"segment": "Mobile"},
                "High",
                85,
                ["Check mobile traffic"],
                [],
                "Anomaly Detection",
            ),
        ]

        steps = prioritize_investigation_steps(insights)

        assert len(steps) > 0
        assert steps[0]["priority"] == "Critical"


class TestExecutiveSummary:
    """Test executive summary generation."""

    def test_executive_summary_generation(self):
        """Test executive summary text generation."""
        insights = [
            Insight(
                "Critical finding",
                {},
                "Critical",
                95,
                ["Action 1"],
                [],
                "Performance",
            ),
            Insight(
                "High finding",
                {},
                "High",
                85,
                ["Action 2"],
                [],
                "Quality",
            ),
        ]

        summary = generate_executive_summary(insights)

        assert "Executive Summary" in summary
        assert "CRITICAL" in summary
        assert "HIGH PRIORITY" in summary

    def test_empty_insights_summary(self):
        """Test summary with no insights."""
        summary = generate_executive_summary([])

        assert "No significant insights" in summary


class TestInsightEvidence:
    """Test insight evidence tracking."""

    def test_insight_with_evidence(self):
        """Test that insights include supporting evidence."""
        insight = Insight(
            finding="Treatment improves conversion",
            evidence={
                "control_rate": 0.05,
                "treatment_rate": 0.08,
                "p_value": 0.001,
            },
            priority="High",
            confidence=95,
            recommendations=["Launch treatment"],
            limitations=["Short test duration"],
            category="Conversion",
        )

        assert insight.evidence["control_rate"] == 0.05
        assert insight.evidence["treatment_rate"] == 0.08
        assert "p_value" in insight.evidence

    def test_insight_with_recommendations(self):
        """Test that insights include actionable recommendations."""
        insight = Insight(
            finding="Mobile conversion dropped",
            evidence={},
            priority="High",
            confidence=80,
            recommendations=[
                "Check mobile traffic sources",
                "Review mobile UX",
                "Test mobile checkout flow",
            ],
            limitations=[],
            category="Performance",
        )

        assert len(insight.recommendations) == 3
        assert "mobile" in insight.recommendations[0].lower()

    def test_insight_with_limitations(self):
        """Test that insights track limitations."""
        insight = Insight(
            finding="Segment analysis",
            evidence={},
            priority="Medium",
            confidence=70,
            recommendations=[],
            limitations=[
                "Small sample size",
                "Correlation not causation",
            ],
            category="Segment",
        )

        assert len(insight.limitations) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
