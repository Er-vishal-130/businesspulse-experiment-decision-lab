"""Advanced visualisations for the A/B Testing Platform.

Provides:
- Funnel chart (Control vs Treatment step-by-step)
- Segment × metric heatmap
- Waterfall chart of conversion contributors
- Trend ribbon (band between group conversion rates over time)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Design tokens (mirror charts.py) ─────────────────────────────────────────
CTRL_COLOR   = "#64748b"
TRT_COLOR    = "#6366f1"
POS_COLOR    = "#22c55e"
NEG_COLOR    = "#f43f5e"
MID_COLOR    = "#94a3b8"
FONT_FAMILY  = "Inter, system-ui, -apple-system, sans-serif"

_LAYOUT_BASE = dict(
    font=dict(family=FONT_FAMILY, size=13, color="#172033"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=56, b=40, l=48, r=20),
    title_font=dict(size=15, color="#172033", family=FONT_FAMILY),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(15,23,42,0.10)",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(15,23,42,0.08)", linecolor="rgba(15,23,42,0.14)"),
    yaxis=dict(gridcolor="rgba(15,23,42,0.08)", linecolor="rgba(15,23,42,0.14)"),
)

def _base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.02), **_LAYOUT_BASE)
    return fig


# ── Funnel chart ──────────────────────────────────────────────────────────────

def funnel_chart(df: pd.DataFrame, steps: list[str] | None = None) -> go.Figure:
    """Return a Plotly funnel chart comparing Control and Treatment groups.

    Parameters
    ----------
    df:
        Filtered experiment DataFrame.
    steps:
        List of binary column names to use as funnel stages.
        Defaults to ['converted'] (single-stage) if only one binary
        column is present.  Multi-stage example: ['pages_viewed_any',
        'bounce_avoided', 'converted'].

    The funnel shows the fraction of each group reaching each stage.
    """
    if steps is None:
        # Auto-detect binary columns for multi-stage funnel
        candidates = ["converted", "bounce"]
        steps = [c for c in candidates if c in df.columns and
                 pd.to_numeric(df[c], errors="coerce").isin([0, 1]).any()]
        if not steps:
            steps = ["converted"] if "converted" in df.columns else []

    if not steps or "group" not in df.columns:
        fig = go.Figure()
        return _base(fig, "Funnel — need group + at least one binary column")

    control = df[df["group"] == "Control"]
    treatment = df[df["group"] == "Treatment"]

    ctrl_values, trt_values = [], []
    for step in steps:
        ctrl_s = pd.to_numeric(control[step], errors="coerce").isin([0, 1])
        trt_s  = pd.to_numeric(treatment[step], errors="coerce").isin([0, 1])
        ctrl_rate = float(pd.to_numeric(control.loc[ctrl_s.index[ctrl_s], step], errors="coerce").mean()) if ctrl_s.any() else 0.0
        trt_rate  = float(pd.to_numeric(treatment.loc[trt_s.index[trt_s],  step], errors="coerce").mean()) if trt_s.any()  else 0.0
        ctrl_values.append(ctrl_rate * 100)
        trt_values.append(trt_rate * 100)

    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name="Control", y=steps, x=ctrl_values,
        textinfo="value+percent initial",
        marker=dict(color=CTRL_COLOR, line=dict(width=1, color="rgba(255,255,255,0.1)")),
    ))
    fig.add_trace(go.Funnel(
        name="Treatment", y=steps, x=trt_values,
        textinfo="value+percent initial",
        marker=dict(color=TRT_COLOR, line=dict(width=1, color="rgba(255,255,255,0.1)")),
    ))
    fig.update_layout(funnelmode="overlay")
    return _base(fig, "Conversion Funnel — Control vs Treatment")


# ---------------------------------------------------------------------------
# Segment × metric heatmap
# ---------------------------------------------------------------------------

def segment_metric_heatmap(
    df: pd.DataFrame,
    segment_col: str,
    metrics: list[str] | None = None,
) -> go.Figure:
    """Heatmap of Z-score normalized metric means for each segment value.

    Parameters
    ----------
    df:
        Experiment DataFrame.
    segment_col:
        Categorical column to use as the Y axis.
    metrics:
        List of numeric metric columns.  Auto-detected if None.
    """
    if metrics is None:
        candidates = ["converted", "revenue", "session_duration", "pages_viewed", "bounce"]
        metrics = [m for m in candidates if m in df.columns]

    if segment_col not in df.columns or not metrics:
        fig = go.Figure()
        fig.update_layout(title="Heatmap — insufficient data")
        return fig

    df2 = df.copy()
    df2["_seg"] = df2[segment_col].astype(str)
    agg = df2.groupby("_seg")[metrics].mean().reset_index()
    agg = agg.rename(columns={"_seg": segment_col})

    matrix = agg[metrics].to_numpy(dtype=float)
    # Z-score normalise each metric column for comparability
    col_means = np.nanmean(matrix, axis=0)
    col_stds = np.nanstd(matrix, axis=0)
    col_stds = np.where(col_stds < 1e-12, 1.0, col_stds)
    z_matrix = (matrix - col_means) / col_stds

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=metrics,
            y=agg[segment_col].tolist(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Z-score"),
            text=np.round(matrix, 3),
            hovertemplate="%{y} | %{x}<br>Z-score: %{z:.2f}<br>Raw mean: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Segment × metric heatmap ({segment_col})",
        xaxis_title="Metric",
        yaxis_title=segment_col.replace("_", " ").title(),
        margin=dict(t=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Waterfall chart of conversion contributors
# ---------------------------------------------------------------------------

def conversion_waterfall_chart(
    segment_rows: list[dict[str, Any]],
    segment_col: str = "segment",
) -> go.Figure:
    """Waterfall chart showing each segment's contribution to the overall lift.

    Parameters
    ----------
    segment_rows:
        List of dicts with keys: segment_value, observed_minus_expected_conversions
        (or absolute_lift).
    segment_col:
        Display name for the segment dimension.
    """
    if not segment_rows:
        fig = go.Figure()
        fig.update_layout(title="Waterfall — no segment data")
        return fig

    labels = [str(r.get("segment_value") or r.get("segment", "")) for r in segment_rows]
    # Try observed_minus_expected_conversions first, fall back to absolute_lift
    values = [
        float(r.get("observed_minus_expected_conversions") or r.get("absolute_lift") or 0.0)
        for r in segment_rows
    ]

    measures = ["relative"] * len(values) + ["total"]
    labels = labels + ["Total lift"]
    values = values + [sum(values)]

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values[:-1]] + ["#5c7cfa"]

    fig = go.Figure(
        go.Waterfall(
            name="Segment contribution",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgba(100,116,139,0.3)"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#5c7cfa"}},
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Conversion contribution waterfall by {segment_col}",
        yaxis_title="Δ conversions vs expected",
        margin=dict(t=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Trend ribbon
# ---------------------------------------------------------------------------

def trend_ribbon_chart(df: pd.DataFrame) -> go.Figure:
    """Conversion rate trend with a ribbon (band) between Control and Treatment.

    Requires a ``date`` column and ``group`` + ``converted`` columns.
    """
    if not {"date", "group", "converted"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(title="Trend ribbon — need date, group, and converted columns")
        return fig

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce", format="mixed")
    df2["converted"] = pd.to_numeric(df2["converted"], errors="coerce")
    df2 = df2[df2["converted"].isin([0, 1]) & df2["date"].notna()]
    if df2.empty:
        fig = go.Figure()
        fig.update_layout(title="Trend ribbon — no valid data after parsing")
        return fig

    df2["date_day"] = df2["date"].dt.date
    trend = (
        df2.groupby(["date_day", "group"])["converted"]
        .mean()
        .reset_index()
        .rename(columns={"converted": "conversion_rate"})
    )
    ctrl = trend[trend["group"] == "Control"].set_index("date_day")["conversion_rate"]
    trt = trend[trend["group"] == "Treatment"].set_index("date_day")["conversion_rate"]
    common_dates = sorted(set(ctrl.index) & set(trt.index))

    if not common_dates:
        fig = go.Figure()
        fig.update_layout(title="Trend ribbon — no overlapping dates between groups")
        return fig

    ctrl_vals = [ctrl[d] for d in common_dates]
    trt_vals = [trt[d] for d in common_dates]

    fig = go.Figure()
    # Fill ribbon between the two lines
    fig.add_trace(
        go.Scatter(
            x=common_dates + common_dates[::-1],
            y=trt_vals + ctrl_vals[::-1],
            fill="toself",
            fillcolor="rgba(92,124,250,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Treatment − Control ribbon",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=common_dates, y=ctrl_vals, mode="lines+markers",
            name="Control", line=dict(color="#64748b", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=common_dates, y=trt_vals, mode="lines+markers",
            name="Treatment", line=dict(color="#5c7cfa", width=2),
        )
    )
    fig.update_layout(
        title="Daily conversion rate ribbon — Control vs Treatment",
        xaxis_title="Date",
        yaxis_title="Conversion rate",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )
    return fig
