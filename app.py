from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.ask_data import answer_question
from modules.bayesian import (
    run_bayesian_ab_test,
    posterior_density_chart,
    credible_interval_chart,
)
from modules.multi_metric import run_multi_metric_test
from modules.sequential import run_sprt, interim_analysis, sprt_boundary_chart
from modules.experiment_history import (
    load_experiment_history,
    compare_experiments,
    timeline_chart,
    parse_history_csv,
)
from visualizations.advanced_charts import (
    funnel_chart,
    segment_metric_heatmap,
    conversion_waterfall_chart,
    trend_ribbon_chart,
)
from modules.business_analytics import categorical_segment_columns, summarize_time_trend
from database.database import (
    init_db,
    insert_experiment,
    list_alert_statuses,
    list_experiments,
    save_analysis_result,
    save_run_summary,
    update_alert_status,
)
from modules.alerts import (
    ALERT_STATUSES,
    DEFAULT_SEVERITY_THRESHOLDS,
    build_alerts,
    filter_and_sort_alerts,
)
from modules.confidence_intervals import calculate_difference_confidence_interval, calculate_rate_confidence_interval
from modules.data_loader import (
    build_upload_metadata,
    ensure_sample_dataset,
    generate_sample_dataset,
    load_kaggle_dataset,
    load_uploaded_dataset,
)
from modules.decision_engine import decide_experiment
from modules.detective import detect_business_anomaly, generate_anomaly_demo_dataset
from modules.effect_size import calculate_effect_size
from modules.guardrails import compute_guardrail_summary, evaluate_guardrail_risk
from modules.hypothesis_testing import run_two_proportion_test
from modules.metrics import build_business_scenario_table, calculate_experiment_health_score, estimate_business_impact, summarize_group_metrics
from modules.power_analysis import minimum_detectable_effect, required_sample_size
from modules.preprocessing import filter_dataset, normalize_dataframe
from modules.report_generator import build_html_report, build_summary_text
from modules.segmentation import analyze_segment
from modules.validator import REQUIRED_COLUMNS, clean_dataframe, validate_dataframe
from visualizations.charts import confidence_interval_chart, conversion_bar_chart, lift_chart, segment_performance_chart

# ── New BusinessPulse AI modules ──────────────────────────────────────────────
from modules.data_quality import profile_dataframe, get_quality_grade, suggest_cleaning_actions, calculate_data_reliability_score
from modules.kpi_intelligence import (
    detect_available_kpis,
    calculate_kpi_values,
    calculate_kpi_trends,
    compare_kpi_periods,
    detect_kpi_anomalies,
    calculate_kpi_health_score,
)
from modules.insights_engine import generate_insights_from_analysis, prioritize_investigation_steps, generate_executive_summary
from modules.investigation import enhanced_root_cause_investigation, drill_down_segment
from modules.funnel_analytics import analyze_conversion_funnel, identify_funnel_bottlenecks, compare_funnels
from modules.cohort_analytics import create_cohort_analysis, calculate_retention_metrics, analyze_repeat_behavior
from modules.query_assistant import answer_query_enhanced
from modules.anomaly_advanced import (
    detect_anomalies_multi_method,
    detect_time_series_anomalies,
    explain_anomaly_detection,
)

init_db()

st.set_page_config(
    page_title="BusinessPulse | Experiment Decision Lab",
    page_icon="🔬",
    layout="wide",
)


DESIGN_THEMES = {
    "Simple": {
        "dark_mode": False,
        "bg": "#f8fafc",
        "panel": "#ffffff",
        "panel2": "#f1f5f9",
        "border": "rgba(15, 23, 42, 0.08)",
        "text": "#111827",
        "muted": "#475569",
        "accent": "#2563eb",
        "accent2": "#1d4ed8",
        "success": "#16a34a",
        "danger": "#dc2626",
        "warn_c": "#d97706",
        "sidebar_bg": "#f8fafc",
        "sidebar_text": "#111827",
    },
    "Premium SaaS": {
        "dark_mode": False,
        "bg": "#f5f7fb",
        "panel": "#ffffff",
        "panel2": "#eef2f8",
        "border": "rgba(15, 23, 42, 0.10)",
        "text": "#172033",
        "muted": "#53627a",
        "accent": "#4f46e5",
        "accent2": "#6366f1",
        "success": "#15803d",
        "danger": "#be123c",
        "warn_c": "#b45309",
        "sidebar_bg": "#111827",
        "sidebar_text": "#f8fafc",
    },
    "Modern SaaS": {
        "dark_mode": True,
        "bg": "#0a0f1e",
        "panel": "#111827",
        "panel2": "#1e2a3a",
        "border": "rgba(99,102,241,0.22)",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "accent": "#6366f1",
        "accent2": "#818cf8",
        "success": "#22c55e",
        "danger": "#f43f5e",
        "warn_c": "#f59e0b",
        "sidebar_bg": "#0d1424",
        "sidebar_text": "#e2e8f0",
    },
    "Dark Premium": {
        "dark_mode": True,
        "bg": "#09090b",
        "panel": "#111318",
        "panel2": "#181b22",
        "border": "rgba(148, 163, 184, 0.16)",
        "text": "#f4f4f5",
        "muted": "#a1a1aa",
        "accent": "#a78bfa",
        "accent2": "#c4b5fd",
        "success": "#4ade80",
        "danger": "#fb7185",
        "warn_c": "#fbbf24",
        "sidebar_bg": "#0f1115",
        "sidebar_text": "#f4f4f5",
    },
    "Executive Light": {
        "dark_mode": False,
        "bg": "#f5f7fb",
        "panel": "#ffffff",
        "panel2": "#edf2f7",
        "border": "rgba(15, 23, 42, 0.10)",
        "text": "#0f172a",
        "muted": "#52607a",
        "accent": "#0f766e",
        "accent2": "#14b8a6",
        "success": "#0f9f6e",
        "danger": "#d14343",
        "warn_c": "#b7791f",
        "sidebar_bg": "#f5f7fb",
        "sidebar_text": "#0f172a",
    },
}


def theme_css() -> None:
    selected_theme = st.session_state.get("design_theme_v2", "Premium SaaS")
    palette = DESIGN_THEMES.get(selected_theme, DESIGN_THEMES["Premium SaaS"])
    dark_mode = palette["dark_mode"]
    bg = palette["bg"]
    panel = palette["panel"]
    panel2 = palette["panel2"]
    border = palette["border"]
    text = palette["text"]
    muted = palette["muted"]
    accent = palette["accent"]
    accent2 = palette["accent2"]
    success = palette["success"]
    danger = palette["danger"]
    warn_c = palette["warn_c"]
    sidebar_bg = palette["sidebar_bg"]
    sidebar_text = palette.get("sidebar_text", text)

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html,body,[class*="css"]{{font-family:'Inter',system-ui,-apple-system,sans-serif!important;}}
        html,body,.stApp,[data-testid="stAppViewContainer"]{{background-color:{bg}!important;color:{text};color-scheme:{"dark" if dark_mode else "light"};}}
        [data-testid="stHeader"]{{background-color:transparent!important;}}
        .stApp p,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,.stApp span,.stApp label,.stApp td,.stApp th,.stApp code,.stApp pre,.stApp .stMarkdown,.stApp .stMarkdownContainer{{color:inherit;}}
        .block-container{{padding:2rem 2.25rem 3rem!important;max-width:1400px;}}
        div[data-testid="stSidebar"]{{background-color:{sidebar_bg}!important;border-right:1px solid {border}!important;}}
        div[data-testid="stSidebar"] p,div[data-testid="stSidebar"] label,div[data-testid="stSidebar"] span,div[data-testid="stSidebar"] h1,div[data-testid="stSidebar"] h2,div[data-testid="stSidebar"] h3{{color:{sidebar_text}!important;}}
        div[data-testid="stSidebar"] .stRadio label{{font-size:0.82rem!important;padding:0.5rem 0.75rem;border-radius:10px;transition:background 0.15s ease,color 0.15s ease;}}
        div[data-testid="stSidebar"] .stRadio label:hover{{background:rgba(148,163,184,0.12);}}
        div[data-testid="metric-container"]{{height:100%;background:{panel}!important;border:1px solid {border}!important;border-radius:12px!important;padding:1rem 1.1rem!important;box-shadow:none;transition:border-color 0.15s ease,background 0.15s ease;}}
        div[data-testid="metric-container"]:hover{{border-color:rgba(148,163,184,0.4)!important;}}
        div[data-testid="metric-container"] label{{color:{muted}!important;font-size:0.7rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.06em!important;}}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"]{{font-size:1.55rem!important;font-weight:700!important;color:{text}!important;}}
        .stTabs [data-baseweb="tab-list"]{{background:{panel}!important;border-radius:10px!important;border:1px solid {border}!important;padding:4px!important;gap:4px!important;}}
        .stTabs [data-baseweb="tab"]{{color:{muted}!important;border-radius:8px!important;font-size:0.82rem!important;font-weight:600!important;padding:0.5rem 0.9rem!important;transition:all 0.15s ease!important;}}
        .stTabs [aria-selected="true"]{{background:{panel2}!important;color:{text}!important;border:1px solid {border}!important;box-shadow:none!important;}}
        .stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button{{background:{accent}!important;color:white!important;border:1px solid {accent}!important;border-radius:8px!important;font-weight:600!important;font-size:0.82rem!important;padding:0.6rem 1.1rem!important;box-shadow:none!important;transition:all 0.15s ease!important;}}
        .stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{{background:{accent2}!important;border-color:{accent2}!important;transform:none!important;filter:none!important;}}
        .stButton>button:active,.stDownloadButton>button:active,[data-testid="stFormSubmitButton"] button:active{{transform:translateY(0)!important;}}
        .stButton>button:disabled,.stDownloadButton>button:disabled,[data-testid="stFormSubmitButton"] button:disabled{{background:{panel2}!important;color:{muted}!important;border-color:{border}!important;box-shadow:none!important;}}
        .stTextInput input,.stNumberInput input,.stDateInput input,.stTextArea textarea{{background:{panel2}!important;border:1px solid {border}!important;border-radius:8px!important;color:{text}!important;}}
        .stTextInput input:focus,.stNumberInput input:focus,.stDateInput input:focus,.stTextArea textarea:focus{{border-color:{accent}!important;box-shadow:0 0 0 3px rgba(99,102,241,0.12)!important;}}
        [data-baseweb="select"]>div{{background:{panel2}!important;border-color:{border}!important;border-radius:8px!important;color:{text}!important;}}
        [data-baseweb="select"] input{{color:{text}!important;}}
        button:focus-visible,input:focus-visible,textarea:focus-visible,[role="tab"]:focus-visible,[role="radio"]:focus-visible{{outline:2px solid {accent2}!important;outline-offset:2px!important;}}
        h1,h2,h3{{letter-spacing:-0.03em;}}
        [data-testid="stCaptionContainer"]{{color:{muted}!important;}}
        details[data-testid="stExpander"]{{border:1px solid {border}!important;border-radius:12px!important;background:{panel}!important;margin-bottom:0.75rem!important;}}
        details[data-testid="stExpander"] summary{{font-weight:600!important;color:{text}!important;padding:0.8rem 1rem!important;}}
        [data-testid="stDataFrame"]{{border:1px solid {border}!important;border-radius:10px!important;overflow:hidden!important;}}
        [data-testid="stPlotlyChart"] .xtick text,[data-testid="stPlotlyChart"] .ytick text,[data-testid="stPlotlyChart"] .legend text,[data-testid="stPlotlyChart"] .infolayer text,[data-testid="stPlotlyChart"] .gtitle{{fill:{text}!important;color:{text}!important;}}
        div[data-testid="stAlert"]{{border-radius:10px!important;border-left-width:4px!important;font-size:0.875rem!important;}}
        .abt-hero{{background:linear-gradient(115deg,{panel} 0%,{panel2} 100%);border:1px solid {border};border-left:4px solid {accent};border-radius:14px;padding:1.55rem 1.7rem;margin-bottom:1.35rem;position:relative;overflow:hidden;box-shadow:0 8px 26px rgba(15,23,42,0.045);}}
        .abt-hero::before{{content:"";position:absolute;top:-55px;right:-35px;width:220px;height:220px;background:radial-gradient(circle,rgba(99,102,241,0.11) 0%,transparent 72%);pointer-events:none;}}
        .abt-hero-brand{{display:flex;align-items:center;gap:0.65rem;margin-bottom:0.9rem;}}
        .abt-brand-mark{{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:10px;background:{accent};color:white!important;font-size:0.78rem;font-weight:800;letter-spacing:-0.04em;}}
        .abt-hero h1{{font-size:1.85rem;font-weight:700;color:{text};margin:0 0 0.4rem;letter-spacing:-0.045em;}}
        .abt-hero p{{color:{muted};font-size:0.92rem;margin:0;max-width:760px;line-height:1.6;}}
        .abt-hero-meta{{display:flex;flex-wrap:wrap;gap:0.45rem;margin-top:1rem;}}
        .abt-hero-meta span{{display:inline-flex;align-items:center;border:1px solid {border};border-radius:999px;background:{panel};padding:0.28rem 0.65rem;color:{muted}!important;font-size:0.72rem;font-weight:600;}}
        .abt-sidebar-brand{{display:flex;align-items:center;gap:0.65rem;padding:0.45rem 0 1rem;margin-bottom:0.65rem;border-bottom:1px solid {border};}}
        .abt-sidebar-brand strong{{display:block;font-size:0.95rem;font-weight:700;line-height:1.2;color:{sidebar_text}!important;}}
        .abt-sidebar-brand small{{display:block;margin-top:0.18rem;font-size:0.67rem;letter-spacing:0.04em;color:{sidebar_text}!important;opacity:0.72;}}
        .abt-dashboard-intro{{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin:0.25rem 0 1rem;}}
        .abt-dashboard-intro h2{{font-size:1.35rem;font-weight:700;color:{text};margin:0.1rem 0 0;letter-spacing:-0.03em;}}
        .abt-dashboard-intro p{{color:{muted};font-size:0.84rem;margin:0.35rem 0 0;}}
        .abt-eyebrow{{color:{accent2};font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;}}
        .abt-decision-card{{height:100%;padding:1.2rem 1.3rem;background:{panel};border:1px solid {border};border-radius:12px;}}
        .abt-decision-card h3{{margin:0.45rem 0;font-size:1.2rem;color:{text};}}
        .abt-decision-card p{{margin:0;color:{muted};font-size:0.9rem;line-height:1.6;}}
        .abt-card-label{{color:{muted};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;}}
        .abt-badge{{display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;margin-right:0.35rem;}}
        .abt-badge-success{{background:rgba(34,197,94,0.12);color:{success};border:1px solid rgba(34,197,94,0.2);}}
        .abt-badge-danger{{background:rgba(244,63,94,0.12);color:{danger};border:1px solid rgba(244,63,94,0.2);}}
        .abt-badge-warning{{background:rgba(245,158,11,0.12);color:{warn_c};border:1px solid rgba(245,158,11,0.2);}}
        .abt-badge-neutral{{background:rgba(148,163,184,0.12);color:{muted};border:1px solid rgba(148,163,184,0.2);}}
        .abt-badge-accent{{background:rgba(99,102,241,0.12);color:{accent2};border:1px solid rgba(99,102,241,0.2);}}
        .abt-section-header{{display:flex;align-items:center;gap:0.6rem;margin:1.2rem 0 0.75rem;padding-bottom:0.45rem;border-bottom:1px solid {border};}}
        .abt-section-header h3{{font-size:1rem;font-weight:600;color:{text};margin:0;}}
        .abt-section-icon{{width:28px;height:28px;background:rgba(99,102,241,0.08);border:1px solid {border};border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;}}
        .abt-divider{{height:1px;background:{border};margin:1.25rem 0;}}
        .abt-stat-pill{{display:inline-flex;align-items:center;gap:0.4rem;background:{panel2};border:1px solid {border};border-radius:8px;padding:0.38rem 0.7rem;font-size:0.8rem;color:{text};margin:0.2rem 0.25rem 0.2rem 0;}}
        .abt-stat-pill strong{{color:{accent2};}}
        .abt-info-box{{background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.18);border-radius:10px;padding:0.8rem 0.9rem;font-size:0.84rem;color:{text};margin:0.4rem 0;}}
        .abt-workflow-steps{{display:flex;flex-wrap:wrap;gap:0.35rem;align-items:center;margin:0.5rem 0;}}
        .abt-workflow-step{{background:{panel2};border:1px solid {border};border-radius:8px;padding:0.3rem 0.65rem;font-size:0.76rem;font-weight:600;color:{accent2};}}
        .abt-workflow-arrow{{color:{muted};font-size:0.8rem;}}
        .abt-health-ring{{font-size:1.6rem;font-weight:800;color:{accent2};}}
        .abt-kpi-card{{background:{panel};border:1px solid {border};border-radius:12px;padding:1rem 1.05rem;text-align:center;box-shadow:none;}}
        .abt-kpi-label{{color:{muted};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.35rem;}}
        .abt-kpi-value{{color:{text};font-size:1.45rem;font-weight:700;line-height:1.2;}}
        .sb-section-label{{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{muted};padding:0.5rem 0 0.2rem;}}
        .sb-data-pill{{display:flex;justify-content:space-between;align-items:center;padding:0.22rem 0;font-size:0.8rem;border-bottom:1px solid rgba(148,163,184,0.12);color:{text};}}
        .sb-data-pill span{{color:{accent2};font-weight:600;}}
        @media (max-width: 900px){{.block-container{{padding:1.2rem 1.2rem 2.5rem!important;}}.abt-hero{{padding:1.3rem 1.4rem;}}}}
        @media (max-width: 600px){{.block-container{{padding:1rem 0.75rem 2rem!important;}}.abt-hero{{padding:1.1rem;border-radius:11px;}}.abt-hero h1{{font-size:1.45rem;}}.abt-section-header{{margin-top:1rem;}}div[data-testid="metric-container"]{{padding:0.85rem!important;}}div[data-testid="metric-container"] div[data-testid="stMetricValue"]{{font-size:1.3rem!important;}}}}
        @media (prefers-reduced-motion: reduce){{*,*::before,*::after{{scroll-behavior:auto!important;transition:none!important;animation:none!important;}}}}
        ::-webkit-scrollbar{{width:5px;height:5px;}}
        ::-webkit-scrollbar-track{{background:transparent;}}
        ::-webkit-scrollbar-thumb{{background:rgba(148,163,184,0.32);border-radius:999px;}}
        ::-webkit-scrollbar-thumb:hover{{background:rgba(148,163,184,0.5);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.get("design_theme_v2") not in DESIGN_THEMES:
    st.session_state.design_theme_v2 = "Premium SaaS"

st.session_state.design_theme_v2 = st.sidebar.selectbox(
    "Design style",
    list(DESIGN_THEMES.keys()),
    index=list(DESIGN_THEMES.keys()).index(st.session_state.design_theme_v2),
)

st.sidebar.caption("Choose the visual style that best matches your workflow.")
theme_css()


def page_title_block() -> None:
    st.markdown(
        """
        <div class="abt-hero">
            <div class="abt-hero-brand">
                <span class="abt-brand-mark">BP</span>
                <span class="abt-eyebrow">BusinessPulse / Experiment Decision Lab</span>
            </div>
            <h1>Make every experiment count.</h1>
            <p>Turn product experiments into clear, defensible decisions with statistical evidence and business context.</p>
            <div class="abt-hero-meta">
                <span>Evidence-led analysis</span>
                <span>Business-aware recommendations</span>
                <span>One workspace, end to end</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




# ── UI helpers ────────────────────────────────────────────────────────────────

def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="abt-badge abt-badge-{kind}">{text}</span>'


def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'''<div class="abt-section-header">
        <div class="abt-section-icon">{icon}</div>
        <h3>{title}</h3></div>''',
        unsafe_allow_html=True,
    )


def stat_pills(*stats) -> None:
    html = "".join(
        f'<div class="abt-stat-pill">{lbl}: <strong>{val}</strong></div>'
        for lbl, val in stats
    )
    st.markdown(html, unsafe_allow_html=True)


def workflow_steps(steps: list) -> None:
    parts = []
    for i, step in enumerate(steps):
        parts.append(f'<div class="abt-workflow-step">{step}</div>')
        if i < len(steps) - 1:
            parts.append('<span class="abt-workflow-arrow">&rarr;</span>')
    st.markdown(f'<div class="abt-workflow-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="abt-info-box">&#x2139;&#xFE0F;&nbsp; {text}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="abt-divider"></div>', unsafe_allow_html=True)

@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    return ensure_sample_dataset()


@st.cache_data
def get_kaggle_dataset() -> pd.DataFrame:
    return load_kaggle_dataset()


@st.cache_data
def get_businesspulse_demo_dataset() -> pd.DataFrame:
    return generate_anomaly_demo_dataset()


APP_TITLE = "Interactive A/B Testing and Business Analytics Platform"
DETECTIVE_SEGMENTS = [
    "device",
    "traffic_source",
    "country",
    "gender",
    "page_type",
    "experiment_id",
]


def detect_segments(data: pd.DataFrame, min_observations: int) -> dict[str, dict]:
    available = [column for column in DETECTIVE_SEGMENTS if column in data.columns]
    return {
        column: detect_business_anomaly(
            data,
            segment_by=column,
            metric_col="converted",
            min_observations=min_observations,
        )
        for column in available
    }


def alerts_for_detections(
    detections: dict[str, dict],
    source_key: str,
    thresholds: dict[str, float],
) -> list[dict]:
    alerts = []
    for dimension, detection in detections.items():
        alerts.extend(
            build_alerts(
                detection,
                source_key=source_key,
                metric_name="Conversion rate",
                thresholds=thresholds,
            )
        )
    return alerts


def data_provenance(source_key: str) -> tuple[str, str]:
    if source_key == "kaggle_zhangluyuan_ab_testing":
        return (
            "Kaggle: A/B testing",
            "Source: zhangluyuan/ab-testing. This dataset has repeated user IDs and inconsistent group/page assignments; independent-row inference is suppressed where repeated IDs are found. Kaggle lists its license as unknown; verify the terms before redistribution.",
        )
    if source_key == "businesspulse_demo":
        return (
            "Synthetic anomaly demo",
            "Entirely synthetic; use only to demonstrate the workflow, never as a business finding.",
        )
    if source_key == "generated":
        return (
            "Generated sample data",
            "Synthetic illustrative data; not a verified business dataset.",
        )
    if source_key == "sample":
        return (
            "Bundled sample data",
            "Project-provided sample; provenance is not independently verified and it is not a verified business dataset.",
        )
    if source_key.startswith("upload:"):
        filename = source_key.split(":", 2)[1]
        return (
            f"User-uploaded file: {filename}",
            "A SHA-256 fingerprint identifies file bytes but does not verify contents or origin. The business provenance of this file has not been independently verified.",
        )
    return (
        "Unknown data source",
        "Data provenance could not be established; treat findings as unverified.",
    )


page_title_block()
sidebar = st.sidebar
sidebar.markdown(
    '<div class="abt-sidebar-brand">'
    '<span class="abt-brand-mark">BP</span>'
    '<span><strong>BusinessPulse</strong><small>EXPERIMENT DECISION LAB</small></span>'
    '</div>',
    unsafe_allow_html=True,
)

data_sources = [
    "Sample dataset",
    "Generated sample",
    "Synthetic anomaly demo",
    "Upload file",
]
if (Path(__file__).resolve().parent / "data" / "ab_data.csv").is_file():
    data_sources.insert(1, "Kaggle A/B dataset (zhangluyuan)")

source = sidebar.radio("Choose data source", data_sources)

if source == "Sample dataset":
    df = get_default_dataset()
    if st.session_state.get("dataset_source") != "sample":
        st.session_state["working_df"] = normalize_dataframe(df)
        st.session_state["dataset_source"] = "sample"
elif source == "Kaggle A/B dataset (zhangluyuan)":
    try:
        df = get_kaggle_dataset()
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the bundled Kaggle dataset: {exc}")
        st.stop()
    if st.session_state.get("dataset_source") != "kaggle_zhangluyuan_ab_testing":
        normalized_kaggle = normalize_dataframe(df)
        st.session_state["working_df"] = normalized_kaggle
        st.session_state["dataset_source"] = "kaggle_zhangluyuan_ab_testing"
    repeated_user_rows = int(df["user_id"].duplicated().sum())
    expected_page = df["group"].astype(str).str.strip().str.casefold().map(
        {"control": "old_page", "treatment": "new_page"}
    )
    mismatched_assignments = int(
        (expected_page != df["page_type"].astype(str).str.strip().str.casefold()).sum()
    )
    st.warning(
        "Kaggle educational dataset: repeated user IDs and group/page assignment "
        f"mismatches are present ({repeated_user_rows:,} repeated-ID rows; "
        f"{mismatched_assignments:,} mismatches). Do not interpret the data as "
        "verified randomized evidence. The platform suppresses independent-row "
        "inference when repeated IDs are detected. Kaggle currently lists the "
        "dataset license as unknown; verify reuse terms before redistribution."
    )
elif source == "Generated sample":
    generated_rows = sidebar.number_input("Rows to generate", min_value=100, max_value=200000, value=5000, step=100)
    generated_seed = sidebar.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    generate_clicked = sidebar.button("Generate sample dataset")
    if generate_clicked or st.session_state.get("dataset_source") != "generated":
        st.session_state["working_df"] = normalize_dataframe(generate_sample_dataset(rows=generated_rows, seed=generated_seed))
        st.session_state["dataset_source"] = "generated"
        if generate_clicked:
            st.success(f"Generated a new sample dataset with {generated_rows:,} rows.")
    df = st.session_state.get("working_df", normalize_dataframe(get_default_dataset()))
elif source == "Synthetic anomaly demo":
    if st.session_state.get("dataset_source") != "businesspulse_demo":
        st.session_state["working_df"] = normalize_dataframe(get_businesspulse_demo_dataset())
        st.session_state["dataset_source"] = "businesspulse_demo"
        st.warning("Demo mode: this synthetic anomaly dataset is for demonstrations only and should not be treated as real business truth.")
    df = st.session_state["working_df"]
else:
    uploaded_file = sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        file_digest = hashlib.sha256(uploaded_bytes).hexdigest()
        upload_key = f"upload:{uploaded_file.name}:{file_digest}"
        if st.session_state.get("dataset_source") != upload_key:
            try:
                df = load_uploaded_dataset(uploaded_file)
            except (ValueError, OSError, pd.errors.ParserError, ImportError) as exc:
                st.error(f"Could not read the uploaded file: {exc}")
                st.stop()
            st.session_state["working_df"] = normalize_dataframe(df)
            st.session_state["dataset_source"] = upload_key
            st.session_state["upload_metadata"] = build_upload_metadata(
                uploaded_file.name,
                uploaded_bytes,
                df,
            )
            st.session_state["upload_metadata_source_key"] = upload_key
            st.success(f"Loaded {uploaded_file.name} ({len(df):,} rows).")
        else:
            df = st.session_state["working_df"]
    else:
        df = get_default_dataset()
        if st.session_state.get("dataset_source") != "sample":
            st.session_state["working_df"] = normalize_dataframe(df)
            st.session_state["dataset_source"] = "sample"
        st.info("Upload a CSV or Excel file in the sidebar. Showing the sample dataset until a file is selected.")

if "working_df" not in st.session_state or st.session_state["working_df"] is None:
    st.session_state["working_df"] = normalize_dataframe(df)

if sidebar.button("Reset dataset to sample"):
    st.session_state["working_df"] = normalize_dataframe(get_default_dataset())
    st.session_state["dataset_source"] = "sample"

if sidebar.button("Clean invalid rows"):
    st.session_state["working_df"] = clean_dataframe(st.session_state["working_df"])

validation = validate_dataframe(st.session_state["working_df"])
provenance_label, provenance_limitations = data_provenance(
    str(st.session_state.get("dataset_source", "unknown"))
)
sidebar.markdown(
    f'<div class="sb-section-label">&#x1F4CA; Data Source</div>'
    f'<div class="sb-data-pill">{provenance_label[:28]}'
    f'<span>{"&#x2713;" if not validation["issues"] else "&#x26A0;"}</span></div>',
    unsafe_allow_html=True,
)
sidebar.caption(provenance_limitations)
active_source_key = str(st.session_state.get("dataset_source", "unknown"))
upload_metadata = st.session_state.get("upload_metadata")
if (
    active_source_key.startswith("upload:")
    and isinstance(upload_metadata, dict)
    and st.session_state.get("upload_metadata_source_key") == active_source_key
):
    with sidebar.expander("Uploaded file metadata"):
        sidebar.write(f"Uploaded at: {upload_metadata['uploaded_at']}")
        sidebar.write(
            f"File size: {upload_metadata['file_size_bytes']:,} bytes; "
            f"rows: {upload_metadata['row_count']:,}; "
            f"columns: {upload_metadata['column_count']:,}"
        )
        sidebar.code(f"SHA-256: {upload_metadata['sha256']}")
        sidebar.caption(
            "The fingerprint can help identify identical file bytes; it does not verify the "
            "file's contents, business origin, or provenance."
        )
        stype_frame = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Inferred type": inferred_type,
                    "Missing values": upload_metadata["missing_values"].get(column, 0),
                }
                for column, inferred_type in upload_metadata["columns"].items()
            ]
        )
        sidebar.dataframe(stype_frame, use_container_width=True, hide_index=True)
sidebar.markdown(
    f'<div class="sb-section-label">&#x1F4CB; Dataset Stats</div>'
    f'<div class="sb-data-pill">Status&nbsp;<span>{validation["status"]}</span></div>'
    f'<div class="sb-data-pill">Rows&nbsp;<span>{validation["rows"]:,}</span></div>'
    f'<div class="sb-data-pill">Columns&nbsp;<span>{validation["columns"]}</span></div>'
    f'<div class="sb-data-pill">Missing&nbsp;<span>{validation["missing_values"]}</span></div>'
    f'<div class="sb-data-pill">Duplicates&nbsp;<span>{validation["duplicate_rows"]}</span></div>',
    unsafe_allow_html=True,
)

saved_experiments = list_experiments()
if saved_experiments:
    sidebar.markdown("### Saved experiments")
    saved_names = [f"#{row['id']} — {row['name']}" for row in saved_experiments[:5]]
    sidebar.write("\n".join(saved_names))

if validation["issues"]:
    sidebar.warning("\n".join(validation["issues"]))

page_options = [
    "Overview",
    "Data Validation",
    "Analytics",
    "Ask Your Data",
    "Statistics",
    "Segmentation",
    "Guardrails",
    "Business Impact",
    "Comparison",
    "BusinessPulse Detective",
    "BusinessPulse AI",
    "Alert Center",
    "KPI Drill-Down",
    "Planner",
    "Simulator",
    "Report",
    "Bayesian Analysis",
    "Multi-Metric",
    "Sequential Test",
    "Experiment History",
    "Advanced Charts",
]
requested_page = st.session_state.pop("requested_page", None)
if requested_page in page_options:
    st.session_state["navigation_page"] = requested_page
page = sidebar.radio(
    "&#x1F9ED; Navigate",
    page_options,
    key="navigation_page",
)

missing_required_columns = [
    column for column in REQUIRED_COLUMNS if column not in st.session_state["working_df"].columns
]
if missing_required_columns and page != "Data Validation":
    st.error(
        "This dataset cannot be analyzed yet because the experiment columns are missing."
    )
    st.info(f"Required for A/B analysis: {', '.join(missing_required_columns)}.")
    st.caption("Other business fields (revenue, date, session_id, device, and traffic_source) are optional.")
    st.stop()

filtered = st.session_state["working_df"].copy()
if page in ["Overview", "Analytics", "Ask Your Data", "Statistics", "Segmentation", "Guardrails", "Business Impact", "Comparison", "BusinessPulse Detective", "BusinessPulse AI", "Alert Center", "KPI Drill-Down", "Planner", "Simulator", "Report", "Bayesian Analysis", "Multi-Metric", "Sequential Test", "Experiment History", "Advanced Charts"]:
    with st.expander("Apply filters"):
        selected_groups = None
        selected_devices = None
        selected_countries = None
        selected_sources = None
        if "group" in filtered:
            group_options = sorted(filtered["group"].dropna().astype(str).unique().tolist())
            selected_groups = st.multiselect("Group", group_options, default=group_options)
        if "device" in filtered:
            device_options = sorted(filtered["device"].dropna().astype(str).unique().tolist())
            selected_devices = st.multiselect("Device", device_options, default=device_options)
        if "country" in filtered:
            country_options = sorted(filtered["country"].dropna().astype(str).unique().tolist())
            selected_countries = st.multiselect("Country", country_options, default=country_options)
        if "traffic_source" in filtered:
            traffic_options = sorted(filtered["traffic_source"].dropna().astype(str).unique().tolist())
            selected_sources = st.multiselect("Traffic Source", traffic_options, default=traffic_options)
        date_min = date_max = None
        if "date" in filtered:
            valid_dates = pd.to_datetime(filtered["date"], errors="coerce", format="mixed").dropna()
            if not valid_dates.empty:
                date_min, date_max = st.date_input(
                    "Date range",
                    value=(valid_dates.min().date(), valid_dates.max().date()),
                )
        age_min = age_max = None
        if "age" in filtered:
            numeric_ages = pd.to_numeric(filtered["age"], errors="coerce").dropna()
            if not numeric_ages.empty:
                age_lower = int(numeric_ages.min())
                age_upper = int(numeric_ages.max())
                if age_lower < age_upper:
                    age_min, age_max = st.slider(
                        "Age range",
                        age_lower,
                        age_upper,
                        (age_lower, age_upper),
                    )
                else:
                    age_min = age_max = age_lower

        filtered = filter_dataset(
            filtered,
            group=selected_groups,
            device=selected_devices,
            country=selected_countries,
            traffic_source=selected_sources,
            date_min=date_min,
            date_max=date_max,
            min_age=age_min,
            max_age=age_max,
        )

if filtered.empty and page != "Data Validation":
    st.warning("The selected filters have removed all data. Please broaden the filters or reset the dataset.")
    st.stop()

summary = summarize_group_metrics(filtered)
control_df = filtered[filtered["group"] == "Control"].copy() if "group" in filtered else pd.DataFrame()
treatment_df = filtered[filtered["group"] == "Treatment"].copy() if "group" in filtered else pd.DataFrame()

alpha = 0.05
alternative = "two-sided"
if page == "Statistics":
    alpha = st.slider("Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)
    alternative_label = st.selectbox(
        "Test type",
        ["Two-sided", "Treatment > Control", "Treatment < Control"],
    )
    alternative = {
        "Two-sided": "two-sided",
        "Treatment > Control": "greater",
        "Treatment < Control": "smaller",
    }[alternative_label]

hypothesis = run_two_proportion_test(control_df, treatment_df, alpha=alpha, alternative=alternative)
control_outcomes = (
    pd.to_numeric(control_df["converted"], errors="coerce")
    if "converted" in control_df
    else pd.Series(dtype=float)
)
treatment_outcomes = (
    pd.to_numeric(treatment_df["converted"], errors="coerce")
    if "converted" in treatment_df
    else pd.Series(dtype=float)
)
control_outcomes = control_outcomes[control_outcomes.isin([0, 1])]
treatment_outcomes = treatment_outcomes[treatment_outcomes.isin([0, 1])]
independence_warning = any(
    "repeated user_id" in warning
    or "repeated session_id" in warning
    for warning in hypothesis["warnings"]
)
minimum_group_n = min(len(control_outcomes), len(treatment_outcomes))
if independence_warning:
    unavailable_interval = {"low": float("nan"), "high": float("nan")}
    control_ci = unavailable_interval
    treatment_ci = unavailable_interval
    diff_ci = unavailable_interval
    size_echo = None
    power_mde = None
else:
    control_ci = calculate_rate_confidence_interval(
        int(control_outcomes.sum()), len(control_outcomes), alpha=alpha
    )
    treatment_ci = calculate_rate_confidence_interval(
        int(treatment_outcomes.sum()), len(treatment_outcomes), alpha=alpha
    )
    diff_ci = calculate_difference_confidence_interval(
        control_df, treatment_df, alpha=alpha
    )
    size_echo = required_sample_size(
        summary["control_conversion"],
        summary["treatment_conversion"],
        alpha=alpha,
        power=0.8,
        two_sided=(alternative == "two-sided"),
    )
    power_mde = minimum_detectable_effect(
        sample_size_per_group=minimum_group_n,
        base_rate=summary["control_conversion"],
        alpha=alpha,
        power=0.8,
        two_sided=(alternative == "two-sided"),
    )

guardrail_summary = compute_guardrail_summary(filtered)
guardrail_risk = evaluate_guardrail_risk(guardrail_summary)
decision, recommendation = decide_experiment(
    summary,
    hypothesis["p_value"],
    alpha=alpha,
    ci_delta=diff_ci,
    guardrail_risk=guardrail_risk,
    total_users=len(filtered),
)

if page == "Overview":
    st.markdown(
        '<div class="abt-dashboard-intro"><div><div class="abt-eyebrow">Experiment workspace / Overview</div>'
        '<h2>Performance at a glance</h2><p>Understand the result, its impact, and the evidence behind it.</p></div></div>',
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Participants", f"{summary['total_users']:,}")
    metric_cols[1].metric("Conversion rate", f"{summary['overall_conversion']:.2%}")
    metric_cols[2].metric("Absolute lift", f"{summary['absolute_lift']:+.2%}")
    metric_cols[3].metric(
        "Statistical evidence",
        f"p = {hypothesis['p_value']:.3g}" if pd.notna(hypothesis["p_value"]) else "N/A",
    )
    available_drilldown_segments = [
        column for column in DETECTIVE_SEGMENTS if column in filtered.columns
    ]
    st.caption(
        "Users are distinct user_id values when provided (otherwise rows); sessions require session_id. "
        "Conversion rate is valid conversions divided by rows with a 0/1 outcome."
    )

    result_col, decision_col = st.columns([1.45, 1], gap="large")
    with result_col:
        section_header("&#x1F4C8;", "Control vs treatment")
        st.plotly_chart(conversion_bar_chart(summary), use_container_width=True)
    with decision_col:
        section_header("&#x2728;", "Recommendation")
        st.markdown(
            '<div class="abt-decision-card"><div class="abt-card-label">Experiment decision</div>'
            f'<h3>{escape(str(decision))}</h3><p>{escape(str(recommendation))}</p></div>',
            unsafe_allow_html=True,
        )
    if pd.notna(hypothesis["p_value"]):
        health_score = calculate_experiment_health_score(summary, hypothesis["p_value"], alpha=alpha, guardrail_risk=guardrail_risk)
        st.metric("Experiment health", f"{health_score}/100")

    if pd.notna(hypothesis["p_value"]) and hypothesis["p_value"] < alpha and summary["absolute_lift"] > 0:
        st.warning("Sequential testing warning: continuously checking and stopping as soon as p < 0.05 can inflate false positives. Use a fixed-horizon test or pre-registered stopping rules.")

    with st.expander("Additional experiment metrics"):
        detail_cols = st.columns(5)
        detail_cols[0].metric("Sessions", f"{summary['total_sessions']:,}" if summary["total_sessions"] is not None else "N/A")
        detail_cols[1].metric("Conversions", f"{summary['total_conversions']:,}")
        detail_cols[2].metric("Revenue", f"${summary['revenue']:,.2f}" if summary["revenue"] is not None else "N/A")
        detail_cols[3].metric("Relative lift", f"{summary['relative_lift']:.2f}%")
        detail_cols[4].metric("Records", f"{summary['total_records']:,}")
        if available_drilldown_segments and st.button(
            "Investigate conversion by segment",
            key="overview_conversion_drilldown",
        ):
            st.session_state["drilldown_dimension"] = available_drilldown_segments[0]
            st.session_state["requested_page"] = "KPI Drill-Down"
            st.rerun()

    st.markdown("### Lift distribution")
    st.plotly_chart(lift_chart(summary), use_container_width=True)
    trend = summarize_time_trend(filtered)
    if trend.empty:
        st.info("Date-wise trends are unavailable: include a valid date column to display them.")
    else:
        st.subheader("Date-wise performance")
        st.plotly_chart(
            px.line(
                trend,
                x="date",
                y="conversion_rate",
                color="group",
                markers=True,
                labels={"date": "Date", "conversion_rate": "Conversion rate", "group": "Experiment group"},
                title="Daily conversion rate by experiment group",
            ),
            use_container_width=True,
        )

elif page == "Data Validation":
    section_header("&#x1F50D;", "Data Validation")
    st.markdown(f"**Dataset Status:** {validation['status']}")
    st.write(f"Rows: {validation['rows']} | Columns: {validation['columns']}")
    if validation["issues"]:
        st.warning("\n".join(validation["issues"]))
    else:
        st.success("No obvious data quality issues found.")
    st.dataframe(filtered.head(20), use_container_width=True)
    quality_profile = profile_dataframe(st.session_state["working_df"])
    reliability = calculate_data_reliability_score(quality_profile)
    quality_columns = st.columns(3)
    quality_columns[0].metric(
        "Data quality",
        f"{quality_profile['overall_quality_score']:.1f}/100",
        get_quality_grade(quality_profile["overall_quality_score"]),
    )
    quality_columns[1].metric(
        "Reliability",
        reliability["overall_reliability"],
    )
    quality_columns[2].metric(
        "Completeness",
        f"{100 - quality_profile['missing_analysis'].get('missing_percentage', 0):.1f}%",
    )
    if quality_profile["column_profiles"]:
        profile_columns = [
            "column_name",
            "inferred_type",
            "null_count",
            "null_percentage",
            "unique_count",
            "has_issues",
        ]
        st.markdown("#### Column quality profile")
        st.dataframe(
            pd.DataFrame(quality_profile["column_profiles"])[profile_columns],
            use_container_width=True,
            hide_index=True,
        )
    cleaning_actions = suggest_cleaning_actions(quality_profile)
    if cleaning_actions:
        st.markdown("#### Suggested cleaning actions")
        st.dataframe(pd.DataFrame(cleaning_actions), use_container_width=True, hide_index=True)
    else:
        st.success("No additional cleaning actions are suggested.")
    if reliability["independence_warnings"]:
        st.warning("\n".join(reliability["independence_warnings"]))

elif page == "Analytics":
    section_header("&#x1F4CA;", "Exploratory Data Analysis")
    if not filtered.empty:
        desc = filtered.describe(include="all").T
        st.dataframe(desc, use_container_width=True)
    st.plotly_chart(conversion_bar_chart(summary), width="stretch")
    st.plotly_chart(lift_chart(summary), width="stretch")
    trend = summarize_time_trend(filtered)
    if trend.empty:
        st.info("Date-wise trends are unavailable: include valid date values to display them.")
    else:
        st.plotly_chart(
            px.line(
                trend,
                x="date",
                y="conversion_rate",
                color="group",
                markers=True,
                labels={"date": "Date", "conversion_rate": "Conversion rate", "group": "Experiment group"},
                title="Conversion rate over time",
            ),
            use_container_width=True,
        )

elif page == "Ask Your Data":
    section_header("&#x1F4AC;", "Ask Your Data")
    st.caption("Answers are rule-based and calculated from the currently filtered dataset. No external AI service is used.")
    quick_tab, enhanced_tab = st.tabs(["Quick answers", "Enhanced query assistant"])
    with quick_tab:
        st.write("Try: “Which device has the highest conversion rate?” or “Did Treatment improve conversion?”")
        question = st.text_input("Ask a business question", placeholder="Which traffic source performs best?")
        if st.button("Get answer", type="primary", key="quick_data_answer"):
            result = answer_question(filtered, question)
            st.markdown(result["answer"])
            if result.get("table") is not None:
                st.dataframe(result["table"], use_container_width=True)
            if result.get("warnings"):
                st.warning("\n".join(result["warnings"]))
    with enhanced_tab:
        st.caption("The enhanced assistant supports KPI, trend, anomaly, data-quality, and top-N query patterns.")
        enhanced_question = st.text_input(
            "Ask an enhanced business question",
            placeholder="Top 3 countries by conversion",
            key="enhanced_data_question",
        )
        if st.button("Analyze question", type="primary", key="enhanced_data_answer"):
            enhanced_result = answer_query_enhanced(filtered, enhanced_question)
            st.markdown(enhanced_result["answer"])
            if enhanced_result.get("data") is not None:
                st.dataframe(enhanced_result["data"], use_container_width=True)
            if enhanced_result.get("methodology"):
                st.caption(enhanced_result["methodology"])
            if enhanced_result.get("confidence") is not None:
                st.metric("Answer confidence", f"{enhanced_result['confidence']}%")
            if enhanced_result.get("suggestions"):
                st.write("Suggested follow-ups:")
                st.write("\n".join(f"- {suggestion}" for suggestion in enhanced_result["suggestions"]))

elif page == "Statistics":
    section_header("&#x1F9EA;", "Hypothesis Testing & Confidence Intervals")
    st.write(f"Null hypothesis: H0: no difference between Control and Treatment")
    st.write(f"Alternative: H1: a difference exists ({alternative})")
    p_value_display = (
        f"{hypothesis['p_value']:.6f}"
        if pd.notna(hypothesis["p_value"])
        else "Not estimable"
    )
    statistic_display = (
        f"{hypothesis['statistic']:.4f}"
        if pd.notna(hypothesis["statistic"])
        else "Not estimable"
    )
    st.write(f"p-value: {p_value_display}")
    st.write(f"Test statistic: {statistic_display}")
    st.write(f"Alpha: {alpha:.3f}")
    st.write(f"Decision: {hypothesis['decision']}")
    st.info(hypothesis["interpretation"])
    st.write(f"Method: {hypothesis['method']}")
    if hypothesis["warnings"]:
        st.warning("\n".join(hypothesis["warnings"]))
    st.caption(
        "The test assumes independent randomized units. Repeated user/session IDs suppress this "
        "test; unique or absent IDs do not prove independence. Clustered or paired designs need "
        "a design-specific analysis."
    )
    if pd.notna(control_ci["low"]) and pd.notna(treatment_ci["low"]):
        st.plotly_chart(confidence_interval_chart(control_ci, treatment_ci), width="stretch")
    else:
        st.info(
            "Confidence intervals are unavailable because repeated identifiers make the "
            "independent-row calculation unsupported."
        )
    st.write(
        f"Treatment minus Control 95% CI: "
        f"[{diff_ci['low']:.3%}, {diff_ci['high']:.3%}]"
        if pd.notna(diff_ci["low"])
        else "Treatment minus Control confidence interval is not estimable without observations in both groups."
    )

    st.subheader("Effect Size")
    effect_size = calculate_effect_size(control_df, treatment_df)
    st.write(f"Difference in proportions: {effect_size['difference_in_proportions']:.4f}")
    st.write(f"Relative risk: {effect_size['relative_risk'] if effect_size['relative_risk'] is not None else 'N/A'}")
    st.write(f"Odds ratio: {effect_size['odds_ratio'] if effect_size['odds_ratio'] is not None else 'N/A'}")

    st.subheader("Power and MDE")
    st.write(
        "Required sample size per group: "
        f"{size_echo:,}" if size_echo is not None else
        "Required sample size per group: Unavailable for these rates and assumptions."
    )
    st.write(
        "Approximate minimum detectable effect: "
        f"{power_mde:.4f}" if power_mde is not None else
        "Approximate minimum detectable effect: Unavailable for these rates, sample sizes, and assumptions."
    )
    st.caption(
        f"The MDE uses the smaller valid arm size (n={minimum_group_n:,}) as an equal-allocation "
        "normal-approximation estimate. It assumes independent binary outcomes and a fixed-horizon design; "
        "repeated or clustered observations need a design-specific method. This is not a post-hoc power claim."
    )

elif page == "Segmentation":
    section_header("&#x1F5C2;&#xFE0F;", "Segment-Level Analysis")
    segment_columns = categorical_segment_columns(filtered)
    if not segment_columns:
        st.info("No categorical segment columns with enough distinct values are available.")
    else:
        segment = st.selectbox("Segment by", segment_columns)
        rows = analyze_segment(filtered, segment)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.plotly_chart(segment_performance_chart(rows), width="stretch")
            st.caption("Segment-level comparisons are exploratory; repeated comparisons increase false-positive risk.")
        else:
            st.info("No segment contains observations in both Control and Treatment.")

elif page == "Guardrails":
    section_header("&#x1F6E1;&#xFE0F;", "Guardrail Metrics")
    available_guardrails = {group: values for group, values in guardrail_summary.items() if values}
    if available_guardrails:
        st.dataframe(pd.DataFrame.from_dict(available_guardrails, orient="index"), use_container_width=True)
    else:
        st.info("No guardrail columns (such as revenue, bounce, error_flag, session_duration, or pages_viewed) are available.")
    if guardrail_risk["flags"]:
        st.warning("\n".join(guardrail_risk["flags"]))
    else:
        st.success("Guardrail metrics do not show major operational risk.")

elif page == "Business Impact":
    section_header("&#x1F4BC;", "Business Impact Estimation")
    monthly_visitors = st.number_input("Monthly visitors", min_value=1000, value=50000, step=1000)
    average_revenue = st.number_input("Average revenue per conversion ($)", min_value=1.0, value=120.0, step=1.0)
    impact = estimate_business_impact(summary, monthly_visitors, average_revenue)
    st.metric("Expected Additional Conversions", f"{impact['additional_conversions']:.0f}")
    st.metric("Expected Additional Revenue", f"${impact['additional_revenue']:.2f}")
    st.caption("These are estimates based on assumptions and should be reviewed with business stakeholders.")

elif page == "Comparison":
    section_header("&#x2696;&#xFE0F;", "Experiment Comparison")
    comparison_rows = [
        {
            "Metric": "Observed conversion rate",
            "Current": summary["control_conversion"],
            "Group": "Control",
        },
        {
            "Metric": "Observed conversion rate",
            "Current": summary["treatment_conversion"],
            "Group": "Treatment",
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)
    st.metric("Treatment minus Control", f"{summary['absolute_lift']:+.2%}")
    st.caption("Comparison uses only the selected rows in the uploaded or sample dataset; no synthetic benchmark is generated.")

elif page == "BusinessPulse Detective":
    section_header("&#x1F575;&#xFE0F;", "BusinessPulse Detective")
    st.caption("Compare independent reference and observed populations, then investigate segment-level deviations.")
    active_source = st.session_state.get("dataset_source", "sample")
    provenance_label, provenance_limitations = data_provenance(str(active_source))
    st.info(f"Data source: {provenance_label}. {provenance_limitations}")
    if active_source == "businesspulse_demo":
        st.warning("Synthetic demo results are for demonstration only and are not business findings.")

    st.markdown(
        """
        <div style="padding: 1rem 1.2rem; border: 1px solid rgba(148,163,184,0.25); border-radius: 12px; background: rgba(15,23,42,0.4);">
            <b>Investigation flow:</b> baseline → observed vs expected → strongest contributor → evidence review
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate_segments = [
        column for column in DETECTIVE_SEGMENTS if column in filtered.columns
    ]
    if not candidate_segments:
        st.info("No supported segment columns are available. Add a categorical field such as device, country, or traffic_source.")
    else:
        chosen_segment = st.selectbox("Investigate segment", candidate_segments, index=0)
        screening_minimum = st.number_input(
            "Minimum valid observations per comparison population",
            min_value=1,
            max_value=10000,
            value=30,
            step=5,
            help="Screening threshold only. Meeting this minimum does not establish adequate statistical power.",
        )
        st.caption(
            "This minimum is a low-volume screening rule, not a power calculation. "
            "Displayed p-values use independent-row assumptions and can be unreliable for repeated or clustered observations."
        )
        detection = detect_business_anomaly(
            filtered,
            segment_by=chosen_segment,
            metric_col="converted",
            min_observations=screening_minimum,
        )
        if detection["warnings"]:
            st.warning("\n".join(detection["warnings"]))
        if detection["top_segment"] is None:
            if not detection["warnings"]:
                st.info("No anomaly signal is available for this dataset and segment.")
            accounting = detection["segment_accounting"]
            if accounting["excluded_segments"]:
                st.markdown("**Excluded segments and reasons**")
                st.dataframe(
                    pd.DataFrame(accounting["excluded_segments"]),
                    use_container_width=True,
                )
        else:
            top = detection["top_segment"]
            if detection["alert_count"]:
                st.success(f"{detection['alert_count']} segment(s) passed the adjusted significance threshold.")
            elif not detection["inferential_valid"]:
                st.warning(
                    "Inferential alerts and significance claims are suppressed because repeated or duplicate "
                    "observations make the independent-row assumption uncertain. Differences remain descriptive."
                )
            else:
                st.info("No statistically supported anomaly passed the adjusted significance threshold.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{top['reference_label']} rate", f"{top['expected_rate']:.2%}")
            c2.metric(f"{top['observed_label']} rate", f"{top['observed_rate']:.2%}")
            c3.metric("Absolute difference", f"{top['absolute_lift']:+.2%}")
            if detection["inferential_valid"]:
                c4.metric("Adjusted p-value", f"{top['adjusted_p_value']:.3g}")
            else:
                c4.metric("Inferential status", "Unavailable")

            if not detection["inferential_valid"]:
                finding_label = "Largest observed deviation (descriptive; inference unavailable)"
            elif top["is_anomaly"]:
                finding_label = "Largest statistically supported deviation"
            else:
                finding_label = "Largest observed deviation (not statistically supported)"
            st.markdown(
                f"**{finding_label}:** **{top['segment']}** in **{chosen_segment}** — "
                f"{top['observed_conversions']}/{top['observations']} conversions "
                f"({top['observed_rate']:.2%}) versus {top['expected_conversions']:.1f} expected "
                f"at the reference rate of {top['expected_rate']:.2%}. "
                f"Difference: {top['absolute_lift']:+.2%}; "
                f"{top['observed_minus_expected_conversions']:+.1f} conversions versus expectation."
            )
            if detection["inferential_valid"]:
                st.caption(
                    "95% Newcombe-Wilson interval for the observed-minus-reference rate difference: "
                    f"[{top['difference_ci_low']:+.2%}, {top['difference_ci_high']:+.2%}]. "
                    "The interval describes uncertainty under independent-row assumptions; it is not adjusted for all segments."
                )
            else:
                st.caption(
                    "Confidence intervals and p-values are omitted because the independent-row "
                    "assumption is not supported by the available data."
                )

            chart_df = detection["rows"][["segment", "observed_rate", "expected_rate"]].copy()
            chart_df = chart_df.rename(
                columns={
                    "segment": chosen_segment,
                    "observed_rate": "Observed",
                    "expected_rate": "Expected",
                }
            )
            chart_df_long = chart_df.melt(
                id_vars=chosen_segment,
                var_name="Measure",
                value_name="Conversion rate",
            )
            st.plotly_chart(
                px.bar(
                    chart_df_long,
                    x=chosen_segment,
                    y="Conversion rate",
                    color="Measure",
                    barmode="group",
                    labels={chosen_segment: chosen_segment.title()},
                    title=f"Observed vs expected conversion by {chosen_segment}",
                ),
                use_container_width=True,
            )
            displayed_rows = detection["rows"].copy()
            if not detection["inferential_valid"]:
                displayed_rows = displayed_rows.drop(
                    columns=[
                        "z_score",
                        "p_value",
                        "test_method",
                        "difference_ci_low",
                        "difference_ci_high",
                        "adjusted_p_value",
                        "is_anomaly",
                    ],
                    errors="ignore",
                )
            st.dataframe(displayed_rows.head(10), use_container_width=True)
            if detection["inferential_valid"]:
                st.caption(
                    f"Reference: {top['reference_label']} (n={top['reference_observations']}); "
                    f"observed segment: {top['observed_label']} (n={top['observations']}). "
                    f"Test: {top['test_method']}; raw p={top['p_value']:.3g}; "
                    f"Bonferroni-adjusted across {len(detection['rows'])} tested segment(s). "
                    "These are observational comparisons, not proof of causation. Repeated users or "
                    "clustered data need a design-specific test."
                )
            else:
                st.caption(
                    "Test statistics and intervals are not shown because dependence could not be "
                    "ruled out; observed counts and rates are descriptive only."
                )
            st.write(detection["summary"])
            accounting = detection["segment_accounting"]
            if detection["baseline_label"] == "Control":
                st.markdown("#### Segment contribution accounting")
                st.write(
                    f"Included segments: {len(accounting['included_segments'])}; "
                    f"included Treatment observations: {accounting['included_treatment_observations']:,} "
                    f"of {accounting['total_treatment_observations']:,}."
                )
                st.caption(
                    "Included categories: "
                    + (", ".join(accounting["included_segments"]) or "none")
                )
                st.write(
                    f"Contribution from included segments: {accounting['included_contribution']:+.2f} "
                    "conversions versus Control expectation."
                )
                st.write(
                    f"Full aggregate Treatment change: {accounting['full_aggregate_change']:+.2f} "
                    "conversions versus Control expectation."
                )
                st.caption(
                    f"Unallocated difference: {accounting['unallocated_difference']:+.2f}. "
                    "This residual includes excluded/missing-category segments and is not redistributed; "
                    "included-segment totals need not reconcile to the full aggregate change."
                )
                if accounting["excluded_segments"]:
                    st.markdown("**Excluded segments and reasons**")
                    st.dataframe(
                        pd.DataFrame(accounting["excluded_segments"]),
                        use_container_width=True,
                    )

elif page == "Alert Center":
    section_header("&#x1F514;", "Smart Alert Center")
    status_saved_message = st.session_state.pop("alert_status_saved_message", None)
    if status_saved_message:
        st.success(status_saved_message)
    st.caption(
        "Alerts are generated only from the existing detector's adjusted-significant findings. "
        "Severity is an operational magnitude band, not a validated risk or business-impact score."
    )
    active_source = str(st.session_state.get("dataset_source", "unknown"))
    provenance_label, provenance_limitations = data_provenance(active_source)
    st.info(f"Data source: {provenance_label}. {provenance_limitations}")

    threshold_defaults = st.session_state.get(
        "alert_severity_thresholds", DEFAULT_SEVERITY_THRESHOLDS.copy()
    )
    threshold_columns = st.columns(3)
    medium_pp = threshold_columns[0].number_input(
        "Low → Medium threshold (percentage points)",
        min_value=0.0,
        max_value=100.0,
        value=float(threshold_defaults["medium"] * 100),
        step=0.5,
        key="alert_medium_threshold_pp",
    )
    high_pp = threshold_columns[1].number_input(
        "Medium → High threshold (percentage points)",
        min_value=0.0,
        max_value=100.0,
        value=float(threshold_defaults["high"] * 100),
        step=0.5,
        key="alert_high_threshold_pp",
    )
    critical_pp = threshold_columns[2].number_input(
        "High → Critical threshold (percentage points)",
        min_value=0.0,
        max_value=100.0,
        value=float(threshold_defaults["critical"] * 100),
        step=0.5,
        key="alert_critical_threshold_pp",
    )
    severity_thresholds = {
        "medium": medium_pp / 100,
        "high": high_pp / 100,
        "critical": critical_pp / 100,
    }
    if not medium_pp < high_pp < critical_pp:
        st.error("Severity thresholds must increase from Medium through Critical.")
        st.stop()
    st.session_state["alert_severity_thresholds"] = severity_thresholds
    st.caption(
        "Configured thresholds are app-level screening choices. They are not universal or scientifically validated cutoffs. "
        "Severity does not replace adjusted significance, uncertainty, or business review. "
        "Bonferroni correction is applied within each detector segment dimension, not jointly across dimensions."
    )

    screening_minimum = st.number_input(
        "Minimum valid observations per comparison population",
        min_value=1,
        max_value=10000,
        value=30,
        step=5,
        key="alert_screening_minimum",
        help="Low-volume screening only; this is not a statistical power analysis.",
    )
    detections = detect_segments(filtered, screening_minimum)
    candidate_alerts = alerts_for_detections(
        detections, active_source, severity_thresholds
    )
    persisted_statuses = list_alert_statuses()
    for alert in candidate_alerts:
        alert["status"] = persisted_statuses.get(alert["alert_id"], "New")
    independence_warning = any(
        "repeated user_id" in warning or "repeated session_id" in warning
        for result in detections.values()
        for warning in result.get("warnings", [])
    )
    duplicate_warning = any(
        "exact duplicate row(s)" in warning
        for result in detections.values()
        for warning in result.get("warnings", [])
    )
    if independence_warning or duplicate_warning:
        st.warning(
            "At least one segment analysis reports repeated user/session identifiers or exact duplicate rows. "
            "The existing detector suppresses inferential alerts when these may represent dependent observations."
        )
    if not candidate_alerts:
        st.info(
            "No detector-supported alerts are available for the current data and filters. "
            "Use KPI Drill-Down to inspect screened segment changes, including exploratory changes."
        )
    else:
        filter_columns = st.columns(4)
        selected_severities = filter_columns[0].multiselect(
            "Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
        )
        selected_statuses = filter_columns[1].multiselect(
            "Status", list(ALERT_STATUSES), default=list(ALERT_STATUSES)
        )
        metric_options = sorted({alert["metric"] for alert in candidate_alerts})
        selected_metrics = filter_columns[2].multiselect(
            "KPI", metric_options, default=metric_options
        )
        segment_options = sorted(
            {f"{alert['segment_column']} = {alert['segment']}" for alert in candidate_alerts}
        )
        selected_segments = filter_columns[3].multiselect(
            "Segment", segment_options, default=segment_options
        )
        sort_columns = st.columns(2)
        sort_by = sort_columns[0].selectbox(
            "Sort by", ["Severity", "Date", "Magnitude"], index=0
        )
        sort_descending = sort_columns[1].toggle(
            "Descending", value=True, key="alert_sort_descending"
        )
        visible_alerts = filter_and_sort_alerts(
            candidate_alerts,
            severities=selected_severities,
            statuses=selected_statuses,
            metrics=selected_metrics,
            segments=selected_segments,
            sort_by=sort_by,
            descending=sort_descending,
        )
        table_rows = [
            {
                "Alert ID": alert["alert_id"],
                "KPI": alert["metric"],
                "Detected (UTC)": alert["detected_at"],
                "Observed": alert["observed_value"],
                "Expected": alert["expected_value"],
                "Absolute change (pp)": alert["absolute_change"] * 100,
                "Relative change (%)": alert["percentage_change"],
                "Segment": f"{alert['segment_column']} = {alert['segment']}",
                "Severity": alert["severity"],
                "Status": alert["status"],
                "Why this severity": alert["severity_rationale"],
                "Evidence": alert["statistical_status"],
            }
            for alert in visible_alerts
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        if not visible_alerts:
            st.info("No alerts match the selected filters.")
        else:
            option_labels = {
                f"{alert['alert_id']} · {alert['severity']} · "
                f"{alert['segment_column']}={alert['segment']}": alert
                for alert in visible_alerts
            }
            preselected_alert_id = st.session_state.get("selected_alert_id")
            preselected_label = next(
                (
                    label
                    for label, alert in option_labels.items()
                    if alert["alert_id"] == preselected_alert_id
                ),
                next(iter(option_labels)),
            )
            if st.session_state.get("selected_alert_label") not in option_labels:
                st.session_state["selected_alert_label"] = preselected_label
            selected_label = st.selectbox(
                "Open alert details",
                list(option_labels),
                key="selected_alert_label",
            )
            selected_alert = option_labels[selected_label]
            st.markdown(f"#### Alert {selected_alert['alert_id']}")
            st.write(
                f"**{selected_alert['metric']}** — "
                f"{selected_alert['segment_column']} = {selected_alert['segment']}"
            )
            st.write(
                f"Observed {selected_alert['observed_value']:.2%} vs expected "
                f"{selected_alert['expected_value']:.2%}; absolute change "
                f"{selected_alert['absolute_change']:+.2%}."
            )
            relative_change = selected_alert["percentage_change"]
            st.write(
                "Relative change: "
                + (
                    f"{relative_change:+.2f}%"
                    if relative_change is not None
                    else "not defined because the baseline is zero or unavailable"
                )
            )
            st.caption(
                f"Severity: {selected_alert['severity']}. Rule: "
                f"{selected_alert['severity_rationale']}. "
                "Adjusted statistical findings are exploratory and do not establish causality."
            )
            evidence = selected_alert["evidence"]
            st.write(
                f"Reference n={evidence['reference_observations']:,}; "
                f"observed n={evidence['observations']:,}; "
                f"observed conversions={evidence['observed_conversions']:,}; "
                f"expected conversions={evidence['expected_conversions']:.2f}; "
                f"adjusted p={evidence['adjusted_p_value']:.4g} "
                f"({evidence['test_method']})."
            )
            status_column, action_column = st.columns([2, 1])
            new_status = status_column.selectbox(
                "Update status",
                list(ALERT_STATUSES),
                index=list(ALERT_STATUSES).index(selected_alert["status"]),
                key=f"alert_status_{selected_alert['alert_id']}",
            )
            if action_column.button(
                "Save status", key=f"save_alert_{selected_alert['alert_id']}"
            ):
                update_alert_status(selected_alert["alert_id"], new_status)
                st.session_state["alert_status_saved_message"] = (
                    f"Alert status saved as {new_status}."
                )
                st.rerun()
            if st.button(
                "Investigate this alert",
                key=f"investigate_alert_{selected_alert['alert_id']}",
            ):
                st.session_state["selected_alert_id"] = selected_alert["alert_id"]
                st.session_state["drilldown_dimension"] = selected_alert["segment_column"]
                st.session_state["drilldown_segment"] = selected_alert["segment"]
                st.session_state["requested_page"] = "KPI Drill-Down"
                st.rerun()

elif page == "KPI Drill-Down":
    section_header("&#x1F52C;", "KPI Drill-Down & Impact Analysis")
    active_source = str(st.session_state.get("dataset_source", "unknown"))
    provenance_label, provenance_limitations = data_provenance(active_source)
    st.info(f"Data source: {provenance_label}. {provenance_limitations}")
    candidate_segments = [
        column for column in DETECTIVE_SEGMENTS if column in filtered.columns
    ]
    if not candidate_segments:
        st.info(
            "No supported segment columns are available for drill-down. "
            "Provide a categorical field such as device, country, or traffic_source."
        )
    else:
        requested_dimension = st.session_state.get("drilldown_dimension")
        if requested_dimension not in candidate_segments:
            requested_dimension = candidate_segments[0]
        chosen_segment = st.selectbox(
            "Segment dimension",
            candidate_segments,
            index=candidate_segments.index(requested_dimension),
            key="drilldown_dimension",
        )
        screening_minimum = st.number_input(
            "Minimum valid observations per comparison population",
            min_value=1,
            max_value=10000,
            value=30,
            step=5,
            key="drilldown_screening_minimum",
            help="Low-volume screening only; this is not a statistical power analysis.",
        )
        st.caption(
            "The existing detector's baseline and tests are reused. Repeated IDs can invalidate "
            "independent-row inference; adjusted segment comparisons remain exploratory, not causal. "
            "Bonferroni adjustment is within this dimension's screened segments, not across dimensions."
        )
        detection = detect_business_anomaly(
            filtered,
            segment_by=chosen_segment,
            metric_col="converted",
            min_observations=screening_minimum,
        )
        if detection["warnings"]:
            st.warning("\n".join(detection["warnings"]))
        rows = detection["rows"]
        if rows.empty:
            st.info(
                "No segment has enough valid observations to calculate a comparison for this dimension."
            )
            accounting = detection["segment_accounting"]
            if accounting["excluded_segments"]:
                st.dataframe(
                    pd.DataFrame(accounting["excluded_segments"]),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            magnitude_rows = rows.assign(
                absolute_change_magnitude=rows["absolute_lift"].abs()
            ).sort_values(
                ["absolute_change_magnitude", "observations"],
                ascending=[False, False],
            )
            largest_change = magnitude_rows.iloc[0]
            st.markdown(
                f"**Largest observed change:** {largest_change['segment']} "
                f"({chosen_segment}) at {largest_change['observed_rate']:.2%} observed "
                f"vs {largest_change['expected_rate']:.2%} expected "
                f"({largest_change['absolute_lift']:+.2%}; n={largest_change['observations']:,})."
            )
            supported_rows = rows[rows["is_anomaly"]]
            if detection["inferential_valid"] and not supported_rows.empty:
                st.write(
                    f"**Adjusted-significant exploratory finding(s):** {len(supported_rows)}. "
                    "Adjusted significance does not establish causality."
                )
            else:
                st.write(
                    "**Statistical significance:** no inferential alert is supported by the current "
                    "detector result (or alerting is suppressed for dependent observations)."
                )

            accounting = detection["segment_accounting"]
            if detection["baseline_label"] == "Control":
                contribution_rows = rows.assign(
                    contribution_magnitude=rows[
                        "observed_minus_expected_conversions"
                    ].abs()
                ).sort_values(
                    ["contribution_magnitude", "observations"],
                    ascending=[False, False],
                )
                largest_contribution = contribution_rows.iloc[0]
                st.markdown(
                    f"**Largest calculated included-segment contribution:** "
                    f"{largest_contribution['segment']} at "
                    f"{largest_contribution['observed_minus_expected_conversions']:+.2f} "
                    "conversions versus Control expectation."
                )
                impact_columns = st.columns(3)
                impact_columns[0].metric(
                    "Included segment contribution",
                    f"{accounting['included_contribution']:+.2f}",
                )
                impact_columns[1].metric(
                    "Full aggregate change",
                    f"{accounting['full_aggregate_change']:+.2f}",
                )
                impact_columns[2].metric(
                    "Unallocated residual",
                    f"{accounting['unallocated_difference']:+.2f}",
                )
                st.caption(
                    "Contribution totals include only screened-in, non-missing categories. "
                    "Excluded categories remain in the displayed residual and are never redistributed."
                )
                if accounting["excluded_segments"]:
                    st.markdown("**Excluded segments and reasons**")
                    st.dataframe(
                        pd.DataFrame(accounting["excluded_segments"]),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info(
                    "Aggregate contribution accounting is unavailable for this leave-one-segment-out "
                    "comparison; only observed-versus-reference differences are shown."
                )

            chart_data = rows[["segment", "observed_rate", "expected_rate"]].melt(
                id_vars="segment",
                var_name="Measure",
                value_name="Conversion rate",
            )
            chart_data["Measure"] = chart_data["Measure"].map(
                {"observed_rate": "Observed", "expected_rate": "Expected"}
            )
            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="segment",
                    y="Conversion rate",
                    color="Measure",
                    barmode="group",
                    title=f"Observed vs expected conversion by {chosen_segment}",
                ),
                use_container_width=True,
            )
            display_columns = [
                "segment",
                "observations",
                "observed_conversions",
                "observed_rate",
                "expected_rate",
                "absolute_lift",
                "relative_lift_pct",
                "observed_minus_expected_conversions",
            ]
            if detection["inferential_valid"]:
                display_columns.extend(["adjusted_p_value", "is_anomaly"])
            st.dataframe(
                magnitude_rows[display_columns],
                use_container_width=True,
                hide_index=True,
            )
            segment_options = list(dict.fromkeys(rows["segment"].astype(str).tolist()))
            requested_segment = st.session_state.get("drilldown_segment")
            if requested_segment not in segment_options:
                requested_segment = str(largest_change["segment"])
            selected_drilldown_segment = st.selectbox(
                "Inspect segment evidence",
                segment_options,
                index=segment_options.index(requested_segment),
                key="drilldown_segment",
            )
            linked_alerts = build_alerts(
                detection,
                source_key=active_source,
                thresholds=st.session_state.get(
                    "alert_severity_thresholds", DEFAULT_SEVERITY_THRESHOLDS
                ),
            )
            linked_alert = next(
                (
                    alert
                    for alert in linked_alerts
                    if alert["segment"] == selected_drilldown_segment
                ),
                None,
            )
            if linked_alert is not None:
                saved_status = list_alert_statuses().get(
                    linked_alert["alert_id"], "New"
                )
                st.info(
                    f"Related alert {linked_alert['alert_id']} — {linked_alert['severity']} — "
                    f"{saved_status}. Its severity rule is magnitude-based and configurable."
                )
                if st.button(
                    "Open related alert in Alert Center",
                    key=f"open_related_{linked_alert['alert_id']}",
                ):
                    st.session_state["selected_alert_id"] = linked_alert["alert_id"]
                    st.session_state["requested_page"] = "Alert Center"
                    st.rerun()
            st.caption(
                "Rows are sorted by absolute observed rate change. A contribution is shown only as "
                "observed minus expected conversions; aggregate accounting is available only for the "
                "Control/Treatment comparison and does not establish causality."
            )

elif page == "Planner":
    section_header("&#x1F4D0;", "Experiment Planner & Sample Size Calculator")
    st.info("Power is the probability of detecting a true effect. MDE is the smallest effect the study is designed to detect.")
    base_col, target_col = st.columns(2)
    baseline_rate = base_col.number_input("Baseline conversion rate", min_value=0.0, max_value=1.0, value=float(summary["control_conversion"]), step=0.001)
    expected_rate = target_col.number_input("Expected treatment conversion rate", min_value=0.0, max_value=1.0, value=float(summary["treatment_conversion"]), step=0.001)

    planner_col1, planner_col2, planner_col3 = st.columns(3)
    planner_alpha = planner_col1.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    planner_power = planner_col2.slider("Desired power", 0.60, 0.99, 0.80, 0.01)
    two_sided = planner_col3.toggle("Two-sided test", value=True)

    required_n = required_sample_size(
        baseline_rate,
        expected_rate,
        alpha=planner_alpha,
        power=planner_power,
        two_sided=two_sided,
    )
    planned_mde = (
        minimum_detectable_effect(
            required_n,
            baseline_rate,
            alpha=planner_alpha,
            power=planner_power,
            two_sided=two_sided,
        )
        if required_n is not None
        else None
    )

    st.metric(
        "Required sample size per group",
        f"{required_n:,}" if required_n is not None else "Unavailable",
    )
    st.metric(
        "Estimated minimum detectable effect",
        f"{planned_mde:.4f}" if planned_mde is not None else "Unavailable",
    )
    st.caption(
        "Approximate normal method for a fixed-horizon, independent two-group binary conversion test "
        "with equal allocation. Boundary rates or a zero target effect are not estimable with this method; "
        "a result here does not verify a study's randomization, independence, or achieved power."
    )
    st.write("Interpretation: increasing sample size reduces uncertainty and makes smaller effects detectable. If the expected lift is small, the study may require many more users.")

elif page == "Simulator":
    section_header("&#x1F3DB;&#xFE0F;", "Business Simulator")
    visitors = st.number_input("Base monthly visitors", min_value=1000, max_value=1000000, value=50000, step=1000)
    lift_rate = st.slider("Expected absolute lift", min_value=0.0, max_value=0.20, value=float(summary["absolute_lift"]), step=0.001)
    revenue_per_conversion = st.number_input("Revenue per conversion ($)", min_value=1.0, max_value=2000.0, value=120.0, step=1.0)
    scaling_factors = [0.5, 1.0, 1.5, 2.0, 3.0]
    scenario_df = build_business_scenario_table(visitors, lift_rate, revenue_per_conversion, scaling_factors)
    st.dataframe(scenario_df, use_container_width=True)
    st.caption("This simulation shows how business value changes across different traffic volumes while holding the expected lift constant.")

elif page == "Report":
    section_header("&#x1F4C4;", "Analytical Report & Export")
    report_source_key = str(st.session_state.get("dataset_source", "unknown"))
    report_source, report_limitations = data_provenance(report_source_key)
    report_upload_metadata = (
        upload_metadata
        if report_source_key.startswith("upload:")
        and isinstance(upload_metadata, dict)
        and st.session_state.get("upload_metadata_source_key") == report_source_key
        else None
    )
    st.info(f"Data source: {report_source}. {report_limitations}")
    summary_text = build_summary_text(summary, hypothesis["p_value"], diff_ci, decision, recommendation)
    summary_text += f"\n\nData source: {report_source}\nProvenance limitation: {report_limitations}"
    if hypothesis["warnings"]:
        summary_text += "\n\nStatistical readiness warnings:\n- " + "\n- ".join(
            hypothesis["warnings"]
        )
    if report_upload_metadata:
        summary_text += (
            f"\nUploaded file: {report_upload_metadata['filename']}"
            f"\nUploaded at (UTC): {report_upload_metadata['uploaded_at']}"
            f"\nRows / columns: {report_upload_metadata['row_count']} / {report_upload_metadata['column_count']}"
            f"\nFile size (bytes): {report_upload_metadata['file_size_bytes']}"
            f"\nSHA-256 (file-byte fingerprint): {report_upload_metadata['sha256']}"
            f"\nMissing-value counts by column: "
            f"{json.dumps(report_upload_metadata['missing_values'], ensure_ascii=False)}"
        )
    report_thresholds = st.session_state.get(
        "alert_severity_thresholds", DEFAULT_SEVERITY_THRESHOLDS
    )
    report_screening_minimum = int(
        st.session_state.get("alert_screening_minimum", 30)
    )
    report_detections = detect_segments(
        filtered, min_observations=report_screening_minimum
    )
    report_alerts = alerts_for_detections(
        report_detections,
        str(st.session_state.get("dataset_source", "unknown")),
        report_thresholds,
    )
    report_statuses = list_alert_statuses()
    for alert in report_alerts:
        alert["status"] = report_statuses.get(alert["alert_id"], "New")
    ordered_report_alerts = filter_and_sort_alerts(
        report_alerts, sort_by="Severity", descending=True
    )
    summary_text += (
        f"\nAlert low-volume screening minimum: {report_screening_minimum} valid observations "
        "per comparison population; this is not a power analysis."
    )
    if ordered_report_alerts:
        summary_text += "\n\nDetector-supported alerts (exploratory):\n"
        summary_text += "\n".join(
            f"{alert['alert_id']} | {alert['severity']} | {alert['metric']} | "
            f"{alert['segment_column']}={alert['segment']} | "
            f"observed={alert['observed_value']:.2%} | "
            f"expected={alert['expected_value']:.2%} | status={alert['status']}"
            for alert in ordered_report_alerts
        )
        summary_text += (
            "\nBonferroni adjustment is within each segment dimension; "
            "the dimensions are not jointly adjusted."
        )
        st.markdown("#### Detector-supported alerts (exploratory)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Alert ID": alert["alert_id"],
                        "KPI": alert["metric"],
                        "Segment": f"{alert['segment_column']} = {alert['segment']}",
                        "Severity": alert["severity"],
                        "Status": alert["status"],
                        "Observed": alert["observed_value"],
                        "Expected": alert["expected_value"],
                        "Rule": alert["severity_rationale"],
                    }
                    for alert in ordered_report_alerts
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Alerts reuse the current detector and filtered dataset. Severity thresholds are operational, "
            "user-configured bands; adjusted significance remains exploratory and does not establish causality. "
            "Correction is per segment dimension, not across the full set of dimensions."
        )
    else:
        summary_text += "\n\nDetector-supported alerts: none for this dataset and current filters."
        st.info("No detector-supported alerts are available for the current data and filters.")
    st.code(summary_text)
    html_report = build_html_report(summary, hypothesis, recommendation, decision)
    if hypothesis["warnings"]:
        warning_items = "".join(
            f"<li>{escape(warning)}</li>" for warning in hypothesis["warnings"]
        )
        html_report += (
            "<section><h2>Statistical readiness warnings</h2>"
            f"<ul>{warning_items}</ul></section>"
        )
    if ordered_report_alerts:
        alert_items = "".join(
            "<li>"
            f"{escape(alert['alert_id'])} — {escape(alert['severity'])} — "
            f"{escape(alert['metric'])} — "
            f"{escape(alert['segment_column'])}={escape(alert['segment'])}; "
            f"observed {alert['observed_value']:.2%}, expected {alert['expected_value']:.2%}; "
            f"status {escape(alert['status'])}. "
            f"Rule: {escape(alert['severity_rationale'])}"
            "</li>"
            for alert in ordered_report_alerts
        )
        alert_html = f"<section><h2>Detector-supported alerts (exploratory)</h2><ul>{alert_items}</ul></section>"
    else:
        alert_html = (
            "<section><h2>Detector-supported alerts</h2>"
            "<p>None for this dataset and current filters.</p></section>"
        )
    html_report += alert_html
    html_report += (
        f"<section><h2>Data provenance</h2><p>{escape(report_source)}</p>"
        f"<p>{escape(report_limitations)}</p>"
        f"<p>Alert screening minimum: {report_screening_minimum} valid observations per "
        "comparison population; this is not a statistical power analysis.</p>"
        "<p>Adjusted segment findings are exploratory and do not establish causality, "
        "statistical power, or independent observations. Bonferroni adjustment is within "
        "each segment dimension; dimensions are not jointly adjusted.</p></section>"
    )
    if report_upload_metadata:
        column_metadata_html = "".join(
            "<li>"
            f"{escape(column)}: {escape(inferred_type)}; "
            f"missing values: {report_upload_metadata['missing_values'].get(column, 0)}"
            "</li>"
            for column, inferred_type in report_upload_metadata["columns"].items()
        )
        html_report += (
            "<section><h2>Uploaded file metadata</h2>"
            f"<p>Filename: {escape(report_upload_metadata['filename'])}</p>"
            f"<p>Upload time (UTC): {escape(report_upload_metadata['uploaded_at'])}</p>"
            f"<p>Rows: {report_upload_metadata['row_count']}; "
            f"columns: {report_upload_metadata['column_count']}; "
            f"file size: {report_upload_metadata['file_size_bytes']} bytes.</p>"
            f"<p>SHA-256 file-byte fingerprint: {escape(report_upload_metadata['sha256'])}. "
            "This fingerprint helps identify identical file bytes; it does not verify file contents or origin.</p>"
            f"<ul>{column_metadata_html}</ul></section>"
        )
    csv_summary = {
        **summary,
        "data_source": report_source,
        "provenance_limitation": report_limitations,
        "statistical_readiness_warnings": json.dumps(
            hypothesis["warnings"], ensure_ascii=False
        ),
        "uploaded_filename": (
            report_upload_metadata["filename"] if report_upload_metadata else None
        ),
        "uploaded_at_utc": (
            report_upload_metadata["uploaded_at"] if report_upload_metadata else None
        ),
        "uploaded_file_size_bytes": (
            report_upload_metadata["file_size_bytes"] if report_upload_metadata else None
        ),
        "uploaded_file_sha256": (
            report_upload_metadata["sha256"] if report_upload_metadata else None
        ),
        "uploaded_file_columns": (
            json.dumps(report_upload_metadata["columns"], ensure_ascii=False)
            if report_upload_metadata
            else None
        ),
        "uploaded_file_missing_values": (
            json.dumps(report_upload_metadata["missing_values"], ensure_ascii=False)
            if report_upload_metadata
            else None
        ),
        "detector_alert_count": len(ordered_report_alerts),
        "alert_screening_minimum": report_screening_minimum,
        "alert_adjustment_scope": (
            "Bonferroni within each segment dimension; dimensions are not jointly adjusted."
        ),
        "detector_alerts": json.dumps(
            [
                {
                    "alert_id": alert["alert_id"],
                    "metric": alert["metric"],
                    "segment_column": alert["segment_column"],
                    "segment": alert["segment"],
                    "severity": alert["severity"],
                    "status": alert["status"],
                    "observed_value": alert["observed_value"],
                    "expected_value": alert["expected_value"],
                    "severity_rationale": alert["severity_rationale"],
                }
                for alert in ordered_report_alerts
            ],
            ensure_ascii=False,
        ),
    }
    st.download_button("Download summary CSV", data=pd.DataFrame([csv_summary]).to_csv(index=False).encode("utf-8"), file_name="ab_test_summary.csv", mime="text/csv")
    st.download_button("Download HTML report", data=html_report.encode("utf-8"), file_name="ab_test_report.html", mime="text/html")

    if st.button("Save experiment to database"):
        experiment_id = insert_experiment("Interactive AB Test", dataset_name=report_source, status="completed", notes=report_limitations)
        save_analysis_result(experiment_id, "control_conversion", summary["control_conversion"])
        save_analysis_result(experiment_id, "treatment_conversion", summary["treatment_conversion"])
        save_analysis_result(experiment_id, "p_value", hypothesis["p_value"])
        save_analysis_result(experiment_id, "absolute_lift", summary["absolute_lift"])
        save_run_summary(experiment_id, summary_text, recommendation)
        st.success(f"Experiment saved with ID {experiment_id}.")

    if saved_experiments:
        st.subheader("Recent saved experiments")
        st.dataframe(pd.DataFrame(saved_experiments).drop(columns=["notes"], errors="ignore"), use_container_width=True)


elif page == "Bayesian Analysis":
    section_header("&#x1F9EE;", "Bayesian A/B Analysis")
    st.caption(
        "Bayesian inference uses your data to update a prior belief about conversion rates. "
        "Results include posterior means, credible intervals, and the probability that Treatment beats Control."
    )
    bayes_col1, bayes_col2 = st.columns(2)
    prior_alpha_val = float(bayes_col1.number_input(
        "Prior alpha (Beta distribution)", min_value=0.01, max_value=100.0, value=1.0, step=0.5,
        help="Alpha parameter of the Beta prior. 1 = uninformative (uniform).",
    ))
    prior_beta_val = float(bayes_col2.number_input(
        "Prior beta (Beta distribution)", min_value=0.01, max_value=100.0, value=1.0, step=0.5,
        help="Beta parameter of the Beta prior. 1 = uninformative (uniform).",
    ))
    credible_mass_pct = st.slider(
        "Credible interval mass (%)", min_value=80, max_value=99, value=95, step=1
    )
    n_mc = int(st.select_slider(
        "Monte-Carlo samples", options=[10_000, 50_000, 100_000, 200_000], value=100_000
    ))
    bayes_result = run_bayesian_ab_test(
        control_df, treatment_df,
        prior_alpha=prior_alpha_val,
        prior_beta=prior_beta_val,
        credible_mass=credible_mass_pct / 100,
        n_samples=n_mc,
    )
    if "error" in bayes_result:
        st.error(bayes_result["error"])
    else:
        b_cols = st.columns(4)
        b_cols[0].metric(
            "Control posterior mean",
            f"{bayes_result['control_posterior_mean']:.3%}",
        )
        b_cols[1].metric(
            "Treatment posterior mean",
            f"{bayes_result['treatment_posterior_mean']:.3%}",
        )
        b_cols[2].metric(
            "P(Treatment > Control)",
            f"{bayes_result['prob_treatment_best']:.1%}",
        )
        b_cols[3].metric(
            "Expected loss (choose Ctrl)",
            f"{bayes_result['expected_loss_if_choose_control']:.4f}",
        )
        st.markdown(
            f"**{credible_mass_pct}% credible interval for Treatment minus Control:** "
            f"[{bayes_result['difference_credible_low']:+.3%}, "
            f"{bayes_result['difference_credible_high']:+.3%}]. "
            f"Posterior mean difference: {bayes_result['difference_mean']:+.3%}."
        )
        st.plotly_chart(posterior_density_chart(bayes_result), width="stretch")
        st.plotly_chart(credible_interval_chart(bayes_result), width="stretch")
        with st.expander("Posterior parameters"):
            st.write(f"Control: Beta({bayes_result['ctrl_posterior_alpha']:.2f}, {bayes_result['ctrl_posterior_beta']:.2f})")
            st.write(f"Treatment: Beta({bayes_result['trt_posterior_alpha']:.2f}, {bayes_result['trt_posterior_beta']:.2f})")
            st.write(f"Expected loss if choosing Treatment: {bayes_result['expected_loss_if_choose_treatment']:.4f}")
            st.write(f"Monte-Carlo samples: {bayes_result['n_mc_samples']:,}")
    if bayes_result.get("warnings"):
        st.warning("\n".join(bayes_result["warnings"]))
    st.caption(
        "The Beta prior Beta(1,1) is uninformative (equivalent to having seen 0 successes and 0 failures). "
        "Increasing prior_alpha/beta encodes prior belief of a higher/lower conversion rate. "
        "Bayesian results are not p-values; a high P(Treatment > Control) is not a guarantee of practical value."
    )

elif page == "Multi-Metric":
    section_header("&#x1F4CA;", "Multi-Metric A/B Testing")
    st.caption(
        "Tests all available outcome metrics simultaneously with Holm-Bonferroni correction "
        "to control the family-wise error rate across outcomes."
    )
    mm_alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="mm_alpha")
    mm_result = run_multi_metric_test(control_df, treatment_df, alpha=mm_alpha)
    if mm_result["warnings"]:
        st.warning("\n".join(mm_result["warnings"]))
    if mm_result["summary_df"].empty:
        st.info("No supported metric columns found. Add revenue, session_duration, pages_viewed, or bounce to your dataset.")
    else:
        def _color_sig(val):
            if val is True:
                return "background-color: rgba(34,197,94,0.2)"
            elif val is False:
                return "background-color: rgba(239,68,68,0.1)"
            return ""
        st.dataframe(
            mm_result["summary_df"].style.apply(
                lambda column: column.map(_color_sig), subset=["significant"]
            ),
            use_container_width=True,
        )
        significant_metrics = [r["metric"] for r in mm_result["rows"] if r.get("significant")]
        if significant_metrics:
            st.success(
                f"Metrics with Holm-adjusted significance at alpha={mm_alpha}: "
                f"{', '.join(significant_metrics)}."
            )
        else:
            st.info(f"No metrics reached Holm-adjusted significance at alpha={mm_alpha}.")
        st.caption(
            "Holm-Bonferroni controls the family-wise error rate within the tested metrics, "
            "not across all possible business comparisons. Binary metrics use a proportion test; "
            "continuous metrics use Mann-Whitney U. Effect size for binary = absolute difference; "
            "for continuous = Cohen's d."
        )

elif page == "Sequential Test":
    section_header("&#x23F1;&#xFE0F;", "Sequential / Interim Analysis")
    st.caption(
        "Check whether it is statistically safe to stop the experiment early. "
        "Two methods are available: SPRT (likelihood ratio) and O'Brien-Fleming alpha spending."
    )
    seq_method = st.radio("Method", ["SPRT (Likelihood Ratio Test)", "O'Brien-Fleming (Interim Analysis)"], horizontal=True)

    if seq_method.startswith("SPRT"):
        st.markdown("**Sequential Probability Ratio Test (SPRT)**")
        sprt_col1, sprt_col2, sprt_col3 = st.columns(3)
        sprt_alpha = sprt_col1.slider("Alpha (type-I error)", 0.01, 0.10, 0.05, 0.01, key="sprt_alpha")
        sprt_beta = sprt_col2.slider("Beta (type-II error)", 0.05, 0.40, 0.20, 0.05, key="sprt_beta")
        auto_h1 = sprt_col3.toggle("Auto-detect H1 delta", value=True)
        if auto_h1:
            h1_delta_val = None
        else:
            h1_delta_val = sprt_col3.number_input(
                "H1 absolute lift (delta)", min_value=0.001, max_value=0.50,
                value=0.02, step=0.001, key="sprt_h1_delta"
            )
        sprt_res = run_sprt(
            control_df, treatment_df,
            h1_delta=h1_delta_val,
            alpha=sprt_alpha,
            beta=sprt_beta,
        )
        if sprt_res.get("warnings"):
            st.warning("\n".join(sprt_res["warnings"]))
        if sprt_res["decision"] != "Not estimable":
            seq_cols = st.columns(4)
            seq_cols[0].metric("Log LR", f"{sprt_res['log_lambda']:.4f}" if pd.notna(sprt_res["log_lambda"]) else "N/A")
            seq_cols[1].metric("Upper boundary", f"{sprt_res['boundary_upper']:.4f}" if pd.notna(sprt_res["boundary_upper"]) else "N/A")
            seq_cols[2].metric("Lower boundary", f"{sprt_res['boundary_lower']:.4f}" if pd.notna(sprt_res["boundary_lower"]) else "N/A")
            seq_cols[3].metric("Safe to stop?", "✅ Yes" if sprt_res["safe_to_stop"] else "🔄 No")
            if sprt_res["safe_to_stop"]:
                st.success(sprt_res["decision"])
            else:
                st.info(sprt_res["decision"])
            st.plotly_chart(sprt_boundary_chart(control_df, treatment_df, sprt_res), width="stretch")
        else:
            st.error(sprt_res.get("decision", "Not estimable"))

    else:
        st.markdown("**O'Brien-Fleming Interim Analysis**")
        planned_n = int(st.number_input(
            "Planned sample size per group (from pre-registration)",
            min_value=50, max_value=1_000_000, value=5000, step=100
        ))
        obf_alpha = st.slider("Overall alpha", 0.01, 0.10, 0.05, 0.01, key="obf_alpha")
        obf_two_sided = st.toggle("Two-sided test", value=True, key="obf_two_sided")
        obf_res = interim_analysis(
            control_df, treatment_df,
            planned_n_per_group=planned_n,
            alpha=obf_alpha,
            two_sided=obf_two_sided,
        )
        if obf_res.get("warnings"):
            st.warning("\n".join(obf_res["warnings"]))
        if obf_res["decision"] != "Not estimable":
            obf_cols = st.columns(4)
            obf_cols[0].metric("Information fraction", f"{obf_res['information_fraction']:.1%}" if pd.notna(obf_res["information_fraction"]) else "N/A")
            obf_cols[1].metric("Z-statistic", f"{obf_res['z_statistic']:.4f}" if pd.notna(obf_res["z_statistic"]) else "N/A")
            obf_cols[2].metric("OBF boundary (z)", f"{obf_res['obf_boundary']:.4f}" if pd.notna(obf_res["obf_boundary"]) else "N/A")
            obf_cols[3].metric("Safe to stop?", "✅ Yes" if obf_res["safe_to_stop"] else "🔄 No")
            if obf_res["safe_to_stop"]:
                st.success(obf_res["decision"])
            else:
                st.info(obf_res["decision"])
        else:
            st.error("Not estimable — ensure both groups have valid binary conversion observations.")
    st.caption(
        "Sequential tests are not a substitute for a pre-registered stopping rule. "
        "Repeated testing without boundary correction inflates false-positive rates. "
        "These tools assume independent binary observations."
    )

elif page == "Experiment History":
    section_header("&#x1F4DA;", "Experiment Registry & History")
    hist_tab1, hist_tab2 = st.tabs(["Saved experiments", "Import CSV history"])
    with hist_tab1:
        hist_df = load_experiment_history()
        if hist_df.empty:
            st.info("No experiments have been saved yet. Use the Report page to save an experiment to the database.")
        else:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if "created_at" in hist_df.columns and len(hist_df) >= 2:
                metric_options = [c for c in hist_df.columns if hist_df[c].dtype in ["float64", "int64"] and c != "id"]
                if metric_options:
                    timeline_metric = st.selectbox("Timeline metric", metric_options)
                    st.plotly_chart(timeline_chart(hist_df, metric_col=timeline_metric), width="stretch")
            st.caption("Experiments are saved via the Report page. Only experiments stored in the current database are shown.")
    with hist_tab2:
        st.markdown(
            "Upload a CSV with columns such as `name`, `control_conversion`, `treatment_conversion`, "
            "`p_value`, `decision`, `created_at` to compare historical runs side-by-side."
        )
        hist_upload = st.file_uploader("Upload history CSV", type=["csv"], key="hist_upload")
        if hist_upload is not None:
            hist_csv_df, hist_warnings = parse_history_csv(hist_upload.getvalue())
            if hist_warnings:
                st.warning("\n".join(hist_warnings))
            if not hist_csv_df.empty:
                st.dataframe(hist_csv_df, use_container_width=True)
                num_cols = [c for c in hist_csv_df.columns if hist_csv_df[c].dtype in ["float64", "int64"]]
                if "created_at" in hist_csv_df.columns and len(hist_csv_df) >= 2 and num_cols:
                    sel_metric = st.selectbox("Chart metric", num_cols, key="hist_csv_metric")
                    st.plotly_chart(timeline_chart(hist_csv_df, metric_col=sel_metric), width="stretch")

elif page == "Advanced Charts":
    section_header("&#x1F4C8;", "Advanced Visualizations")
    adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(
        ["Funnel", "Segment Heatmap", "Waterfall", "Trend Ribbon"]
    )
    with adv_tab1:
        st.markdown("#### Conversion Funnel")
        st.caption("Shows what fraction of each group reaches each binary milestone.")
        binary_cols = [c for c in filtered.columns if filtered[c].nunique() <= 2 and
                       pd.to_numeric(filtered[c], errors="coerce").isin([0, 1]).any()]
        funnel_steps = st.multiselect(
            "Select funnel steps (in order)",
            binary_cols,
            default=[c for c in ["converted", "bounce"] if c in binary_cols],
        )
        st.plotly_chart(funnel_chart(filtered, steps=funnel_steps or None), width="stretch")

    with adv_tab2:
        st.markdown("#### Segment × Metric Heatmap")
        st.caption("Z-score normalised heatmap — positive (blue) values are above the column mean.")
        seg_options = [c for c in ["device", "traffic_source", "country", "gender", "page_type"] if c in filtered.columns]
        if seg_options:
            heatmap_seg = st.selectbox("Segment column", seg_options, key="heatmap_seg")
            numeric_options = [c for c in ["converted", "revenue", "session_duration", "pages_viewed", "bounce"] if c in filtered.columns]
            heatmap_metrics = st.multiselect("Metrics to include", numeric_options, default=numeric_options, key="heatmap_metrics")
            st.plotly_chart(
                segment_metric_heatmap(filtered, heatmap_seg, metrics=heatmap_metrics or None),
                use_container_width=True,
            )
        else:
            st.info("No categorical segment columns found.")

    with adv_tab3:
        st.markdown("#### Conversion Contribution Waterfall")
        st.caption("Shows each segment's contribution to the overall conversion lift versus the Control baseline.")
        wf_segs = [c for c in ["device", "traffic_source", "country", "gender"] if c in filtered.columns]
        if wf_segs:
            wf_col = st.selectbox("Segment dimension", wf_segs, key="waterfall_seg")
            from modules.detective import detect_business_anomaly
            wf_detection = detect_business_anomaly(filtered, segment_by=wf_col, metric_col="converted", min_observations=10)
            if wf_detection["rows"].empty:
                st.info("Not enough data per segment for waterfall chart.")
            else:
                wf_rows = wf_detection["rows"].rename(columns={"segment": "segment_value"}).to_dict("records")
                st.plotly_chart(conversion_waterfall_chart(wf_rows, segment_col=wf_col), width="stretch")
        else:
            st.info("No categorical segment columns found.")

    with adv_tab4:
        st.markdown("#### Daily Trend Ribbon")
        st.caption("Shaded band between Control and Treatment daily conversion rates.")
        if "date" in filtered.columns:
            st.plotly_chart(trend_ribbon_chart(filtered), width="stretch")
        else:
            st.info("No 'date' column found. Add a date column to your dataset to see this chart.")

elif page == "BusinessPulse AI":
    section_header("&#x1F4A1;", "BusinessPulse AI Analytics")
    st.caption(
        "Evidence-based KPI monitoring, insights, investigation, funnel, cohort, and anomaly tools. "
        "Results are computed locally from the selected dataset."
    )
    ai_tabs = st.tabs(
        ["Insights", "KPI Intelligence", "Root Cause", "Funnel", "Cohort", "Advanced Anomalies"]
    )

    with ai_tabs[0]:
        kpi_availability = detect_available_kpis(filtered)
        kpi_values = calculate_kpi_values(filtered, kpi_availability)
        insights = generate_insights_from_analysis(
            summary,
            hypothesis,
            kpi_values=kpi_values,
        )
        st.markdown("#### Executive summary")
        st.text(generate_executive_summary(insights))
        if insights:
            st.markdown("#### Evidence-backed findings")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Priority": insight.priority,
                            "Confidence": f"{insight.confidence:.0f}%",
                            "Category": insight.category,
                            "Finding": insight.finding,
                        }
                        for insight in insights
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            for index, insight in enumerate(insights, start=1):
                with st.expander(f"{index}. {insight.finding}"):
                    st.write("**Evidence**")
                    st.json(insight.evidence)
                    st.write("**Recommendations**")
                    st.write("\n".join(f"- {item}" for item in insight.recommendations))
                    if insight.limitations:
                        st.caption("Limitations: " + "; ".join(insight.limitations))
            investigation_steps = prioritize_investigation_steps(insights)
            if investigation_steps:
                st.markdown("#### Prioritized next steps")
                st.dataframe(
                    pd.DataFrame(investigation_steps),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No actionable insight is available for this dataset and current analysis.")

    with ai_tabs[1]:
        kpi_availability = detect_available_kpis(filtered)
        kpi_values = calculate_kpi_values(filtered, kpi_availability)
        available_kpis = [
            name for name, result in kpi_values.items()
            if result["status"] == "available"
        ]
        if available_kpis:
            kpi_health = calculate_kpi_health_score(kpi_values)
            st.metric(
                "KPI coverage health",
                f"{kpi_health['overall_score']:.1f}/100",
                kpi_health["health_grade"],
            )
            metric_columns = st.columns(4)
            for index, name in enumerate(available_kpis):
                metric_columns[index % len(metric_columns)].metric(
                    name,
                    kpi_values[name]["formatted_value"],
                )
            selected_kpi = st.selectbox("Set a target for KPI health", available_kpis)
            current_value = kpi_values[selected_kpi]["value"]
            target_value = st.number_input(
                f"Target for {selected_kpi}",
                value=float(current_value) if pd.notna(current_value) else 0.0,
                key="businesspulse_kpi_target",
            )
            target_health = calculate_kpi_health_score(
                kpi_values, targets={selected_kpi: target_value}
            )
            st.metric(
                "Target achievement",
                f"{target_health['target_achievement_score']:.0f}%",
                selected_kpi,
            )
        else:
            st.info("No KPI can be calculated from the current columns.")

        if "date" in filtered.columns:
            frequency_label = st.selectbox(
                "Trend frequency",
                ["Daily", "Weekly", "Monthly"],
                key="businesspulse_kpi_frequency",
            )
            frequency = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[frequency_label]
            trends = calculate_kpi_trends(filtered, kpi_availability, frequency=frequency)
            if trends:
                trend_kpi = st.selectbox(
                    "KPI trend",
                    list(trends),
                    key="businesspulse_kpi_trend",
                )
                st.plotly_chart(
                    px.line(
                        trends[trend_kpi],
                        x="date_period",
                        y="value",
                        markers=True,
                        title=f"{trend_kpi} — {frequency_label.lower()} trend",
                    ),
                    use_container_width=True,
                )
                anomaly_method = st.selectbox(
                    "Trend anomaly method",
                    ["zscore", "iqr"],
                    format_func=lambda value: value.upper(),
                    key="businesspulse_kpi_anomaly_method",
                )
                kpi_anomalies = detect_kpi_anomalies(trends, method=anomaly_method)
                anomaly_rows = [
                    {"KPI": name, **anomaly}
                    for name, values in kpi_anomalies.items()
                    for anomaly in values
                ]
                if anomaly_rows:
                    st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No KPI trend anomalies were detected with the selected method.")
            else:
                st.info("No date-based KPI trends are available for this dataset.")
            comparisons = compare_kpi_periods(filtered, kpi_values)
            if comparisons:
                st.markdown("#### First-half vs second-half KPI comparison")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"KPI": name, **comparison}
                            for name, comparison in comparisons.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Add a date column to enable KPI trends and period comparisons.")

    with ai_tabs[2]:
        investigation_dimensions = categorical_segment_columns(filtered)
        if not investigation_dimensions:
            st.info("No categorical dimensions are available for root-cause analysis.")
        else:
            selected_dimensions = st.multiselect(
                "Dimensions to investigate",
                investigation_dimensions,
                default=investigation_dimensions[:3],
                key="businesspulse_investigation_dimensions",
            )
            if selected_dimensions:
                root_cause = enhanced_root_cause_investigation(
                    filtered,
                    dimensions=selected_dimensions,
                )
                if root_cause["status"] == "success":
                    st.markdown("#### Dimension impact ranking")
                    st.dataframe(
                        pd.DataFrame(root_cause["dimension_rankings"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown("#### Highest-contribution segments")
                    st.dataframe(
                        pd.DataFrame(root_cause["top_segments"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown("#### Investigation timeline")
                    st.dataframe(
                        pd.DataFrame(root_cause["investigation_timeline"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown("#### Recommended next steps")
                    st.dataframe(
                        pd.DataFrame(root_cause["next_steps"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    primary_dimension = st.selectbox(
                        "Drill into dimension",
                        selected_dimensions,
                        key="businesspulse_drill_dimension",
                    )
                    primary_segments = filtered[primary_dimension].dropna().unique().tolist()
                    if primary_segments:
                        primary_segment = st.selectbox(
                            "Drill into segment",
                            primary_segments,
                            key="businesspulse_drill_segment",
                        )
                        secondary_dimensions = [
                            dimension for dimension in selected_dimensions
                            if dimension != primary_dimension
                        ]
                        if secondary_dimensions:
                            drilldown = drill_down_segment(
                                filtered,
                                primary_dimension,
                                primary_segment,
                                secondary_dimensions,
                            )
                            st.write(
                                f"Sub-segment analysis for {primary_segment} "
                                f"({drilldown['segment_size']:,} records)"
                            )
                            for dimension, result in drilldown["sub_analysis"].items():
                                with st.expander(dimension.replace("_", " ").title()):
                                    st.dataframe(
                                        pd.DataFrame(result.get("segments", [])),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                else:
                    st.warning(root_cause.get("message", "Root-cause analysis is unavailable."))
            else:
                st.info("Select at least one dimension to start the investigation.")

    with ai_tabs[3]:
        binary_columns = []
        for column in filtered.columns:
            if column in {"user_id", "session_id", "group", "date"}:
                continue
            values = pd.to_numeric(filtered[column], errors="coerce").dropna()
            if not values.empty and values.isin([0, 1]).all():
                binary_columns.append(column)
        if not binary_columns:
            st.info("No binary funnel-step columns were found. Add columns with 0/1 stage outcomes.")
        else:
            preferred_steps = [
                column for column in
                ["viewed", "visited", "clicked", "added_to_cart", "initiated_checkout", "converted"]
                if column in binary_columns
            ]
            default_steps = preferred_steps or binary_columns[:2]
            funnel_steps = st.multiselect(
                "Funnel steps (in order)",
                binary_columns,
                default=default_steps,
                key="businesspulse_funnel_steps",
            )
            group_options = [
                column for column in ["group", "device", "traffic_source", "country"]
                if column in filtered.columns
            ]
            group_choice = st.selectbox(
                "Break down funnel by",
                ["Overall"] + group_options,
                key="businesspulse_funnel_group",
            )
            if funnel_steps:
                funnel_result = analyze_conversion_funnel(
                    filtered,
                    funnel_steps=funnel_steps,
                    group_by=None if group_choice == "Overall" else group_choice,
                )
                if funnel_result["status"] == "success":
                    overall_metrics = funnel_result["overall_metrics"]
                    st.dataframe(
                        pd.DataFrame(overall_metrics["step_metrics"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    bottlenecks = identify_funnel_bottlenecks(overall_metrics)
                    if bottlenecks:
                        st.markdown("#### Funnel bottlenecks")
                        st.dataframe(
                            pd.DataFrame(bottlenecks),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.success("No funnel stage exceeds the default drop-off threshold.")
                    if funnel_result.get("segment_metrics"):
                        st.markdown("#### Segment funnel comparison")
                        segment_rows = []
                        for segment, metrics in funnel_result["segment_metrics"].items():
                            for step in metrics["step_metrics"]:
                                segment_rows.append({"Segment": segment, **step})
                        st.dataframe(
                            pd.DataFrame(segment_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    if {"Control", "Treatment"}.issubset(set(filtered["group"].astype(str))):
                        control_funnel = analyze_conversion_funnel(
                            control_df, funnel_steps=funnel_steps
                        )["overall_metrics"]
                        treatment_funnel = analyze_conversion_funnel(
                            treatment_df, funnel_steps=funnel_steps
                        )["overall_metrics"]
                        comparison = compare_funnels(control_funnel, treatment_funnel)
                        if comparison["status"] == "success":
                            st.markdown("#### Control vs Treatment stages")
                            st.dataframe(
                                pd.DataFrame(comparison["step_comparisons"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                else:
                    st.warning(funnel_result.get("message", "Funnel analysis failed."))

    with ai_tabs[4]:
        cohort_options = {"Monthly": "M", "Weekly": "W"}
        cohort_label = st.selectbox(
            "Cohort period",
            list(cohort_options),
            key="businesspulse_cohort_period",
        )
        retention_periods = st.slider(
            "Retention periods to show",
            min_value=1,
            max_value=24,
            value=12,
            key="businesspulse_retention_periods",
        )
        if {"user_id", "date"}.issubset(filtered.columns):
            cohort_result = create_cohort_analysis(
                filtered,
                cohort_period=cohort_options[cohort_label],
                retention_periods=retention_periods,
            )
            if cohort_result["status"] == "success":
                cohort_table = cohort_result["cohort_table"]
                st.markdown("#### Cohort retention (%)")
                st.dataframe(cohort_table, use_container_width=True)
                retention_summary = calculate_retention_metrics(cohort_table)
                st.json(retention_summary)
            else:
                st.info(cohort_result.get("message", "Cohort analysis is unavailable."))
            repeat_result = analyze_repeat_behavior(filtered)
            if repeat_result["status"] == "success":
                repeat_columns = st.columns(4)
                repeat_columns[0].metric(
                    "Converting users", f"{repeat_result['total_converting_users']:,}"
                )
                repeat_columns[1].metric(
                    "One-time users", f"{repeat_result['one_time_users']:,}"
                )
                repeat_columns[2].metric(
                    "Repeat users", f"{repeat_result['repeat_users']:,}"
                )
                repeat_columns[3].metric(
                    "Repeat rate", f"{repeat_result['repeat_rate']:.1%}"
                )
            else:
                st.info(repeat_result.get("message", "Repeat behavior is unavailable."))
        else:
            st.info("Cohort and repeat behavior analysis require user_id and date columns.")

    with ai_tabs[5]:
        anomaly_dimensions = categorical_segment_columns(filtered)
        if anomaly_dimensions:
            anomaly_dimension = st.selectbox(
                "Segment dimension",
                anomaly_dimensions,
                key="businesspulse_anomaly_dimension",
            )
            anomaly_methods = st.multiselect(
                "Detection methods",
                ["zscore", "iqr", "isolation_forest"],
                default=["zscore", "iqr"],
                key="businesspulse_anomaly_methods",
            )
            minimum_observations = st.number_input(
                "Minimum observations per segment",
                min_value=1,
                max_value=10000,
                value=30,
                step=5,
                key="businesspulse_anomaly_minimum",
            )
            if anomaly_methods:
                anomaly_result = detect_anomalies_multi_method(
                    filtered,
                    segment_by=anomaly_dimension,
                    methods=anomaly_methods,
                    min_observations=minimum_observations,
                )
                for warning in anomaly_result["warnings"]:
                    st.warning(warning)
                if anomaly_result["status"] == "error":
                    st.error("No selected anomaly method could analyze the current data.")
                for method, method_result in anomaly_result["anomalies_by_method"].items():
                    st.markdown(f"#### {method.replace('_', ' ').title()}")
                    st.caption(explain_anomaly_detection(method_result))
                    anomalies = method_result.get("anomalies", [])
                    if anomalies:
                        st.dataframe(
                            pd.DataFrame(anomalies),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No segment anomalies were detected by this method.")
                if anomaly_result["consensus_anomalies"]:
                    st.markdown("#### Consensus anomalies")
                    st.dataframe(
                        pd.DataFrame(anomaly_result["consensus_anomalies"]),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("Add a categorical dimension such as device or country to detect segment anomalies.")

        if "date" in filtered.columns and "converted" in filtered.columns:
            st.markdown("#### Time-series anomalies")
            time_method = st.selectbox(
                "Time-series method",
                ["rolling", "changepoint"],
                format_func=lambda value: (
                    "Rolling baseline" if value == "rolling" else "Change-point"
                ),
                key="businesspulse_timeseries_method",
            )
            window_size = st.number_input(
                "Rolling window (days)",
                min_value=2,
                max_value=90,
                value=7,
                step=1,
                key="businesspulse_timeseries_window",
                disabled=time_method != "rolling",
            )
            timeseries_result = detect_time_series_anomalies(
                filtered,
                method=time_method,
                window_size=window_size,
            )
            for warning in timeseries_result.get("warnings", []):
                st.warning(warning)
            st.caption(explain_anomaly_detection(timeseries_result))
            if timeseries_result.get("anomalies"):
                st.dataframe(
                    pd.DataFrame(timeseries_result["anomalies"]),
                    use_container_width=True,
                    hide_index=True,
                )
            elif not timeseries_result.get("warnings"):
                st.info("No time-series anomalies were detected.")
        else:
            st.info("Add date and converted columns to enable time-series anomaly detection.")

divider()
fr_l, fr_r = st.columns([2, 1])
with fr_l:
    section_header("&#x1F3C1;", "Final Recommendation")
    dec_col = "#22c55e" if "Launch" in decision else ("#f43f5e" if "Pause" in decision else "#94a3b8")
    st.markdown(
        f'''<div style="background:rgba(17,24,39,0.85);border:1px solid {dec_col};'''
        f'''border-left:4px solid {dec_col};border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.75rem;">'''
        f'''<div style="color:{dec_col};font-weight:700;font-size:0.75rem;text-transform:uppercase;'''
        f'''letter-spacing:0.07em;margin-bottom:0.4rem;">Decision</div>'''
        f'''<div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;">{decision}</div></div>''',
        unsafe_allow_html=True,
    )
    info_box(recommendation)
with fr_r:
    if pd.notna(hypothesis["p_value"]):
        hs2 = calculate_experiment_health_score(summary, hypothesis["p_value"], alpha=alpha, guardrail_risk=guardrail_risk)
        st.markdown(
            f'''<div class="abt-kpi-card" style="padding:1.5rem;">'''
            f'''<div class="abt-kpi-label">Experiment Health</div>'''
            f'''<div class="abt-health-ring">{hs2}/100</div></div>''',
            unsafe_allow_html=True,
        )
st.caption(
    "&#x26A0;&#xFE0F; Statistical significance alone does not guarantee business value. "
    "Always review sample size, confidence intervals, effect size, and guardrails before launching."
)

if __name__ == "__main__":
    init_db()
