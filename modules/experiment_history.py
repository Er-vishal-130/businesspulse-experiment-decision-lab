"""Experiment history and comparison utilities.

Reads saved experiments from the database and provides:
- Side-by-side comparison of multiple past runs.
- A timeline chart of key metrics across experiments.
- CSV import of external historical run summaries.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database.database import list_experiments


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def load_experiment_history() -> pd.DataFrame:
    """Return all saved experiments as a DataFrame."""
    rows = list_experiments()
    if not rows:
        return pd.DataFrame(
            columns=["id", "name", "dataset_name", "status", "created_at", "notes"]
        )
    return pd.DataFrame(rows)


def compare_experiments(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a tidy comparison table from a list of experiment dicts.

    Each dict should have keys: name, control_conversion, treatment_conversion,
    absolute_lift, relative_lift, p_value, decision.
    """
    if not experiments:
        return pd.DataFrame()

    rows = []
    for exp in experiments:
        rows.append(
            {
                "Experiment": exp.get("name", "—"),
                "Control conv.": exp.get("control_conversion"),
                "Treatment conv.": exp.get("treatment_conversion"),
                "Absolute lift": exp.get("absolute_lift"),
                "Relative lift (%)": exp.get("relative_lift"),
                "p-value": exp.get("p_value"),
                "Decision": exp.get("decision", "—"),
                "Created": exp.get("created_at", "—"),
            }
        )
    return pd.DataFrame(rows)


def timeline_chart(history_df: pd.DataFrame, metric_col: str = "absolute_lift") -> go.Figure:
    """Return a Plotly line chart of a chosen metric across saved experiments."""
    if history_df.empty or metric_col not in history_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No experiment history to display.")
        return fig

    fig = px.line(
        history_df.sort_values("created_at"),
        x="created_at",
        y=metric_col,
        text="name",
        markers=True,
        title=f"Experiment timeline — {metric_col.replace('_', ' ').title()}",
        labels={"created_at": "Created", metric_col: metric_col.replace("_", " ").title()},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(t=60))
    return fig


def parse_history_csv(csv_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Parse a user-uploaded history CSV.

    Expected columns (case-insensitive): name, control_conversion,
    treatment_conversion, p_value, decision, created_at.

    Returns (parsed_df, list_of_warnings).
    """
    import io

    warnings: list[str] = []
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        return pd.DataFrame(), [f"Could not parse CSV: {exc}"]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"name"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(
            f"CSV is missing recommended column(s): {', '.join(sorted(missing))}. "
            "Some comparisons may not be available."
        )

    numeric_cols = [
        "control_conversion", "treatment_conversion",
        "absolute_lift", "relative_lift", "p_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        warnings.append("The uploaded CSV contains no rows.")

    return df, warnings
