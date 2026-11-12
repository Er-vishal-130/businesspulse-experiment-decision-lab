"""
Automated Insights Engine

Generates evidence-based business insights with transparent methodology:
- Automatic finding generation from analytical results
- Evidence linking and verification
- Priority and confidence scoring
- Actionable recommendations
- Limitation tracking
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class Insight:
    """Structured insight with evidence and recommendations."""

    def __init__(
        self,
        finding: str,
        evidence: dict[str, Any],
        priority: str,
        confidence: float,
        recommendations: list[str],
        limitations: list[str],
        category: str,
    ):
        self.finding = finding
        self.evidence = evidence
        self.priority = priority
        self.confidence = confidence
        self.recommendations = recommendations
        self.limitations = limitations
        self.category = category

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding,
            "evidence": self.evidence,
            "priority": self.priority,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "limitations": self.limitations,
            "category": self.category,
        }


def generate_insights_from_analysis(
    summary: dict,
    hypothesis: dict,
    anomaly_results: dict = None,
    kpi_values: dict = None,
    segment_analysis: dict = None,
) -> list[Insight]:
    """
    Generate comprehensive insights from analytical results.

    Args:
        summary: Experiment summary metrics
        hypothesis: Hypothesis test results
        anomaly_results: Anomaly detection results
        kpi_values: KPI calculation results
        segment_analysis: Segment-level analysis results

    Returns:
        List of Insight objects ordered by priority
    """
    insights = []

    # Generate conversion insights
    if summary and "treatment_conversion" in summary:
        insights.extend(_generate_conversion_insights(summary, hypothesis))

    # Generate KPI insights
    if kpi_values:
        insights.extend(_generate_kpi_insights(kpi_values))

    # Generate anomaly insights
    if anomaly_results:
        insights.extend(_generate_anomaly_insights(anomaly_results))

    # Generate segment insights
    if segment_analysis:
        insights.extend(_generate_segment_insights(segment_analysis))

    # Sort by priority
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    insights.sort(key=lambda x: priority_order.get(x.priority, 4))

    return insights


def _generate_conversion_insights(summary: dict, hypothesis: dict) -> list[Insight]:
    """Generate insights about conversion rate changes."""
    insights = []

    control_rate = summary.get("control_conversion", 0)
    treatment_rate = summary.get("treatment_conversion", 0)
    absolute_lift = summary.get("absolute_lift", 0)
    relative_lift = summary.get("relative_lift", 0)
    p_value = hypothesis.get("p_value")

    # Significant improvement
    if p_value is not None and p_value < 0.05 and absolute_lift > 0:
        confidence = min(95, (1 - p_value) * 100)

        if relative_lift > 20:
            priority = "High"
            finding = (
                f"Treatment shows substantial conversion improvement: "
                f"{treatment_rate:.2%} vs {control_rate:.2%} "
                f"({relative_lift:+.1f}% relative lift)"
            )
        else:
            priority = "Medium"
            finding = (
                f"Treatment shows moderate conversion improvement: "
                f"{treatment_rate:.2%} vs {control_rate:.2%} "
                f"({relative_lift:+.1f}% relative lift)"
            )

        insights.append(Insight(
            finding=finding,
            evidence={
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "absolute_lift": absolute_lift,
                "relative_lift_pct": relative_lift,
                "p_value": p_value,
                "statistical_significance": "Yes (p < 0.05)",
            },
            priority=priority,
            confidence=confidence,
            recommendations=[
                "Verify guardrail metrics show no degradation",
                "Calculate business impact with actual traffic volumes",
                "Consider gradual rollout to validate results",
                "Monitor for novelty effects in first 2-4 weeks",
            ],
            limitations=[
                "Statistical significance does not guarantee business value",
                "Effect may vary across segments not analyzed",
                "Long-term impact may differ from short-term test results",
            ],
            category="Conversion Performance",
        ))

    # Significant decline
    elif p_value is not None and p_value < 0.05 and absolute_lift < 0:
        priority = "Critical" if abs(relative_lift) > 10 else "High"
        finding = (
            f"Treatment shows conversion decline: "
            f"{treatment_rate:.2%} vs {control_rate:.2%} "
            f"({relative_lift:.1f}% relative change)"
        )

        insights.append(Insight(
            finding=finding,
            evidence={
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "absolute_lift": absolute_lift,
                "relative_lift_pct": relative_lift,
                "p_value": p_value,
                "statistical_significance": "Yes (p < 0.05)",
            },
            priority=priority,
            confidence=min(95, (1 - p_value) * 100),
            recommendations=[
                "Do NOT launch treatment",
                "Investigate root cause of decline",
                "Review treatment implementation for technical issues",
                "Analyze segment-level performance for specific problem areas",
            ],
            limitations=[
                "Decline is statistically detected but root cause requires further investigation",
            ],
            category="Conversion Performance",
        ))

    # Inconclusive result
    elif p_value is not None and p_value >= 0.05:
        sample_size = summary.get("total_users", 0)

        if sample_size < 1000:
            priority = "Medium"
            finding = "Test is inconclusive - insufficient sample size"
            recommendations = [
                f"Current sample size ({sample_size:,}) may be too small",
                "Run power analysis to determine required sample size",
                "Consider extending test duration",
            ]
        else:
            priority = "Low"
            finding = (
                f"No statistically significant difference detected "
                f"(p = {p_value:.3f})"
            )
            recommendations = [
                "Treatment and Control perform similarly",
                "Consider implementing treatment if no increased cost/complexity",
                "Alternatively, test a bolder variation",
            ]

        insights.append(Insight(
            finding=finding,
            evidence={
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "p_value": p_value,
                "sample_size": sample_size,
            },
            priority=priority,
            confidence=50,
            recommendations=recommendations,
            limitations=[
                "Lack of significance does not prove no effect exists",
                "Subtle effects may exist but require larger sample",
            ],
            category="Statistical Evidence",
        ))

    return insights


def _generate_kpi_insights(kpi_values: dict) -> list[Insight]:
    """Generate insights about KPI performance."""
    insights = []

    # Check for unavailable critical KPIs
    unavailable_kpis = [
        name for name, info in kpi_values.items()
        if info["status"] == "unavailable" and "Revenue" in name
    ]

    if unavailable_kpis:
        insights.append(Insight(
            finding=f"Critical KPIs unavailable: {', '.join(unavailable_kpis)}",
            evidence={
                "unavailable_kpis": unavailable_kpis,
                "reason": "Missing required data columns",
            },
            priority="Medium",
            confidence=100,
            recommendations=[
                "Add revenue tracking to dataset for complete analysis",
                "Revenue impact cannot be estimated without revenue data",
            ],
            limitations=[],
            category="Data Quality",
        ))

    return insights


def _generate_anomaly_insights(anomaly_results: dict) -> list[Insight]:
    """Generate insights from anomaly detection."""
    insights = []

    if "consensus_anomalies" in anomaly_results:
        consensus = anomaly_results["consensus_anomalies"]

        if consensus:
            for anomaly in consensus[:3]:  # Top 3 consensus anomalies
                segment = anomaly.get("segment", "Unknown")
                observed = anomaly.get("observed_rate", 0)
                expected = anomaly.get("expected_rate", 0)
                deviation = anomaly.get("deviation", 0)
                methods = anomaly.get("methods_agreeing", [])

                priority = "High" if len(methods) >= 3 else "Medium"

                finding = (
                    f"Anomaly detected in {segment}: "
                    f"observed {observed:.2%} vs expected {expected:.2%} "
                    f"(deviation: {deviation:+.2%})"
                )

                insights.append(Insight(
                    finding=finding,
                    evidence={
                        "segment": segment,
                        "observed_rate": observed,
                        "expected_rate": expected,
                        "deviation": deviation,
                        "detection_methods": methods,
                        "consensus_strength": len(methods),
                    },
                    priority=priority,
                    confidence=min(90, 50 + len(methods) * 15),
                    recommendations=[
                        f"Investigate {segment} for data quality issues",
                        f"Review {segment} traffic sources and user behavior",
                        "Check for technical issues affecting this segment",
                        "Compare against historical performance for this segment",
                    ],
                    limitations=[
                        "Anomaly detection identifies unusual patterns, not causes",
                        "Multiple detection methods agree, but correlation ≠ causation",
                    ],
                    category="Anomaly Detection",
                ))

    return insights


def _generate_segment_insights(segment_analysis: dict) -> list[Insight]:
    """Generate insights from segment-level analysis."""
    insights = []

    if "rows" in segment_analysis and not segment_analysis["rows"].empty:
        rows = segment_analysis["rows"]

        # Identify best performing segment
        if "observed_rate" in rows.columns:
            best_segment = rows.loc[rows["observed_rate"].idxmax()]

            insights.append(Insight(
                finding=(
                    f"Best performing segment: {best_segment.get('segment', 'Unknown')} "
                    f"with {best_segment.get('observed_rate', 0):.2%} conversion rate"
                ),
                evidence={
                    "segment": str(best_segment.get("segment", "Unknown")),
                    "conversion_rate": float(best_segment.get("observed_rate", 0)),
                    "observations": int(best_segment.get("observations", 0)),
                },
                priority="Low",
                confidence=70,
                recommendations=[
                    "Analyze characteristics of high-performing segment",
                    "Consider targeting similar audiences",
                    "Test variations optimized for this segment",
                ],
                limitations=[
                    "Segment performance may vary over time",
                    "Success factors may not transfer to other segments",
                ],
                category="Segment Performance",
            ))

    return insights


def prioritize_investigation_steps(insights: list[Insight]) -> list[dict[str, str]]:
    """
    Generate prioritized investigation steps based on insights.

    Returns:
        List of investigation steps with action, reason, and priority
    """
    steps = []

    for insight in insights[:5]:  # Top 5 insights
        if insight.category == "Conversion Performance" and insight.priority in ["Critical", "High"]:
            if "decline" in insight.finding.lower():
                steps.append({
                    "action": f"Investigate root cause of conversion decline",
                    "reason": insight.finding,
                    "priority": insight.priority,
                    "next_steps": insight.recommendations,
                })
            else:
                steps.append({
                    "action": "Validate treatment for production launch",
                    "reason": insight.finding,
                    "priority": insight.priority,
                    "next_steps": insight.recommendations,
                })

        elif insight.category == "Anomaly Detection":
            segment = insight.evidence.get("segment", "Unknown")
            steps.append({
                "action": f"Investigate anomaly in {segment}",
                "reason": insight.finding,
                "priority": insight.priority,
                "next_steps": insight.recommendations,
            })

    return steps


def generate_executive_summary(insights: list[Insight]) -> str:
    """
    Generate executive summary from insights.

    Returns:
        Plain text executive summary
    """
    if not insights:
        return "No significant insights generated from current analysis."

    summary_parts = []

    # Count by priority
    priority_counts = {}
    for insight in insights:
        priority_counts[insight.priority] = priority_counts.get(insight.priority, 0) + 1

    # Summary header
    summary_parts.append("Executive Summary")
    summary_parts.append("=" * 50)
    summary_parts.append("")

    # Key findings
    critical = [i for i in insights if i.priority == "Critical"]
    high = [i for i in insights if i.priority == "High"]

    if critical:
        summary_parts.append("CRITICAL FINDINGS:")
        for insight in critical:
            summary_parts.append(f"• {insight.finding}")
        summary_parts.append("")

    if high:
        summary_parts.append("HIGH PRIORITY FINDINGS:")
        for insight in high[:3]:  # Top 3
            summary_parts.append(f"• {insight.finding}")
        summary_parts.append("")

    # Recommendations
    all_recommendations = []
    for insight in insights[:5]:
        all_recommendations.extend(insight.recommendations[:2])

    if all_recommendations:
        summary_parts.append("TOP RECOMMENDATIONS:")
        for i, rec in enumerate(all_recommendations[:5], 1):
            summary_parts.append(f"{i}. {rec}")
        summary_parts.append("")

    # Summary statistics
    summary_parts.append(f"Total insights generated: {len(insights)}")
    for priority, count in sorted(priority_counts.items(), key=lambda x: ["Critical", "High", "Medium", "Low"].index(x[0])):
        summary_parts.append(f"  {priority}: {count}")

    return "\n".join(summary_parts)


def format_insight_for_display(insight: Insight) -> str:
    """
    Format a single insight for display.

    Returns:
        Formatted string representation
    """
    lines = []

    lines.append(f"Priority: {insight.priority} | Confidence: {insight.confidence}%")
    lines.append(f"Category: {insight.category}")
    lines.append("")
    lines.append(f"Finding: {insight.finding}")
    lines.append("")

    if insight.evidence:
        lines.append("Evidence:")
        for key, value in insight.evidence.items():
            if isinstance(value, float):
                if "rate" in key.lower() or "pct" in key.lower():
                    lines.append(f"  • {key}: {value:.2%}")
                else:
                    lines.append(f"  • {key}: {value:.4f}")
            else:
                lines.append(f"  • {key}: {value}")
        lines.append("")

    if insight.recommendations:
        lines.append("Recommendations:")
        for rec in insight.recommendations:
            lines.append(f"  • {rec}")
        lines.append("")

    if insight.limitations:
        lines.append("Limitations:")
        for lim in insight.limitations:
            lines.append(f"  • {lim}")

    return "\n".join(lines)
