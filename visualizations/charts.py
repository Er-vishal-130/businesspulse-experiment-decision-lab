"""Core chart library with a unified dark/light-aware design system."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

# ── Design tokens ────────────────────────────────────────────────────────────
CTRL_COLOR  = "#64748b"   # slate-500  — Control group
TRT_COLOR   = "#6366f1"   # indigo-500 — Treatment group
POS_COLOR   = "#22c55e"   # green-500  — positive lift
NEG_COLOR   = "#f43f5e"   # rose-500   — negative lift
WARN_COLOR  = "#f59e0b"   # amber-500
MID_COLOR   = "#94a3b8"   # slate-400

FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"

_LAYOUT_BASE = dict(
    font=dict(family=FONT_FAMILY, size=13, color="#172033"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=52, b=40, l=48, r=20),
    title_font=dict(size=15, color="#172033", family=FONT_FAMILY),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(15,23,42,0.10)",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="rgba(15,23,42,0.08)",
        linecolor="rgba(15,23,42,0.14)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(15,23,42,0.08)",
        linecolor="rgba(15,23,42,0.14)",
        tickfont=dict(size=11),
    ),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.02), **_LAYOUT_BASE)
    return fig


# ── Charts ───────────────────────────────────────────────────────────────────

def conversion_bar_chart(summary: dict) -> go.Figure:
    labels = ["Control", "Treatment"]
    values = [
        summary["control_conversion"] if summary["control_users"] else None,
        summary["treatment_conversion"] if summary["treatment_users"] else None,
    ]
    colors = [CTRL_COLOR, TRT_COLOR]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.08)", width=1),
                cornerradius=8,
            ),
            text=[f"{v:.2%}" if v is not None else "No data" for v in values],
            textposition="outside",
            textfont=dict(size=13, color="#172033"),
            width=0.45,
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _apply_base(fig, "Conversion Rate — Control vs Treatment")


def lift_chart(summary: dict) -> go.Figure:
    lift = summary["absolute_lift"]
    color = POS_COLOR if lift >= 0 else NEG_COLOR
    fig = go.Figure(
        go.Bar(
            x=["Absolute Lift"],
            y=[lift],
            marker=dict(
                color=color,
                line=dict(color="rgba(255,255,255,0.08)", width=1),
                cornerradius=8,
            ),
            text=[f"{lift:+.3%}"],
            textposition="outside",
            textfont=dict(size=14, color="#172033"),
            width=0.3,
        )
    )
    fig.update_yaxes(tickformat=".1%")
    return _apply_base(fig, "Absolute Conversion Uplift  (Treatment − Control)")


def confidence_interval_chart(control_ci: dict, treatment_ci: dict) -> go.Figure:
    fig = go.Figure()

    for label, ci, color in [
        ("Control",   control_ci,   CTRL_COLOR),
        ("Treatment", treatment_ci, TRT_COLOR),
    ]:
        mid = (ci["low"] + ci["high"]) / 2
        fig.add_trace(
            go.Scatter(
                x=[ci["low"], ci["high"]],
                y=[label, label],
                mode="lines",
                line=dict(color=color, width=6),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mid],
                y=[label],
                mode="markers",
                marker=dict(color="#172033", size=10, symbol="circle",
                            line=dict(color=color, width=2)),
                showlegend=False,
                hovertemplate=f"{label} mean: %{{x:.3%}}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=ci["low"], y=label,
            text=f"{ci['low']:.2%}",
            showarrow=False, xanchor="right",
            font=dict(size=10, color=MID_COLOR),
        )
        fig.add_annotation(
            x=ci["high"], y=label,
            text=f"{ci['high']:.2%}",
            showarrow=False, xanchor="left",
            font=dict(size=10, color=MID_COLOR),
        )

    fig.update_xaxes(tickformat=".1%")
    return _apply_base(fig, "95% Wilson Confidence Intervals")


def segment_performance_chart(segment_rows: list[dict]) -> go.Figure:
    if not segment_rows:
        return go.Figure()

    rows_sorted = sorted(segment_rows, key=lambda r: r["absolute_lift"])
    segments = [r["segment_value"] for r in rows_sorted]
    lifts    = [r["absolute_lift"]  for r in rows_sorted]
    colors   = [POS_COLOR if v >= 0 else NEG_COLOR for v in lifts]

    fig = go.Figure(
        go.Bar(
            x=lifts,
            y=segments,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.06)", width=1),
                cornerradius=5,
            ),
            text=[f"{v:+.2%}" for v in lifts],
            textposition="outside",
            textfont=dict(size=11),
        )
    )
    fig.add_vline(x=0, line_color=MID_COLOR, line_dash="dot", line_width=1)
    fig.update_xaxes(tickformat=".1%")
    return _apply_base(fig, "Segment Absolute Lift vs Control")
