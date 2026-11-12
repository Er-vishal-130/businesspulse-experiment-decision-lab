from __future__ import annotations

import pandas as pd


def build_summary_text(summary: dict, p_value: float, ci_delta: dict, decision: str, recommendation: str) -> str:
    p_value_text = f"{p_value:.4f}" if pd.notna(p_value) else "Not estimable"
    ci_low = ci_delta.get("low")
    ci_high = ci_delta.get("high")
    ci_text = (
        f"[{ci_low:.3%}, {ci_high:.3%}]"
        if pd.notna(ci_low) and pd.notna(ci_high)
        else "Not estimable under the available design assumptions"
    )
    lines = [
        "Experiment Summary",
        "------------------",
        f"Control conversion rate: {summary['control_conversion']:.2%}",
        f"Treatment conversion rate: {summary['treatment_conversion']:.2%}",
        f"Absolute lift: {summary['absolute_lift']:.3%}",
        f"Relative lift: {summary['relative_lift']:.2f}%",
        f"p-value: {p_value_text}",
        f"95% CI for difference: {ci_text}",
        f"Decision: {decision}",
        recommendation,
    ]
    return "\n".join(lines)


def export_summary_csv(summary: dict, path: str) -> None:
    frame = pd.DataFrame([summary])
    frame.to_csv(path, index=False)


def build_html_report(summary: dict, hypothesis: dict, recommendation: str, decision: str) -> str:
    p_value = hypothesis.get("p_value")
    p_value_text = f"{p_value:.4f}" if pd.notna(p_value) else "Not estimable"
    html = f"""
    <html>
      <head><title>AB Test Report</title></head>
      <body>
        <h1>AB Test Experiment Report</h1>
        <p><strong>Decision:</strong> {decision}</p>
        <p><strong>Recommendation:</strong> {recommendation}</p>
        <h2>Summary</h2>
        <ul>
          <li>Control conversion: {summary['control_conversion']:.2%}</li>
          <li>Treatment conversion: {summary['treatment_conversion']:.2%}</li>
          <li>Absolute lift: {summary['absolute_lift']:.3%}</li>
          <li>Relative lift: {summary['relative_lift']:.2f}%</li>
          <li>p-value: {p_value_text}</li>
        </ul>
      </body>
    </html>
    """
    return html
