"""Rule-based NLP question answering for the A/B Testing Platform.

Answers are calculated exclusively from the supplied DataFrame.
No external AI or LLM service is used.

Supported question patterns (case-insensitive, keyword matching):
  Conversion / experiment
    1.  "Which device has the highest conversion rate?"
    2.  "Which traffic source performs best?"
    3.  "Did Treatment improve conversion?"
    4.  "Compare Control and Treatment"
    5.  "What is the overall conversion rate?"
    6.  "What is the conversion rate for [Control|Treatment]?"
    7.  "Which country converts best?"
    8.  "Which gender converts best?"
    9.  "What is the bounce rate?"
    10. "Which device has the lowest bounce rate?"

  Revenue
    11. "What is the average revenue per user?"
    12. "Which segment has the highest revenue?"
    13. "Which segment has the lowest revenue?"
    14. "What is the total revenue?"
    15. "What is the revenue per conversion?"

  Sessions / engagement
    16. "What is the average session duration?"
    17. "Which device has the longest session?"
    18. "How many pages are viewed on average?"

  Temporal
    19. "Which day of the week has the highest conversion?"
    20. "Which week has the highest conversion?"
    21. "What is the trend over time?"

  Top-N
    22. "Top 3 countries by conversion"
    23. "Bottom 3 devices by revenue"

  General
    24. "Key findings" / "Top findings"
    25. "How many users are in the experiment?"
    26. "What percentage of users converted?"
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from modules.hypothesis_testing import run_two_proportion_test
from modules.metrics import calculate_conversion_rate


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _binary_valid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").pipe(lambda s: s[s.isin([0, 1])])


def _segment_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = df.loc[df[column].notna()].copy()
    frame["segment_value"] = frame[column].astype(str)
    frame["converted"] = pd.to_numeric(frame["converted"], errors="coerce")
    frame = frame[frame["converted"].isin([0, 1])]
    aggs: dict[str, Any] = {
        "observations": ("converted", "size"),
        "conversions": ("converted", "sum"),
        "conversion_rate": ("converted", "mean"),
    }
    if "revenue" in frame.columns:
        frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
        aggs["total_revenue"] = ("revenue", lambda s: s.sum(min_count=1))
        aggs["revenue_per_observation"] = ("revenue", "mean")
        aggs["revenue_observations"] = ("revenue", "count")
    if "session_duration" in frame.columns:
        frame["session_duration"] = pd.to_numeric(frame["session_duration"], errors="coerce")
        aggs["avg_session_duration"] = ("session_duration", "mean")
    return frame.groupby("segment_value", as_index=False, observed=True).agg(**aggs)


def _conversion_answer(df: pd.DataFrame, column: str, label: str, highest: bool = True) -> dict:
    if column not in df.columns:
        return {"answer": f"This dataset has no {column} column."}
    table = _segment_table(df, column)
    if table.empty:
        return {"answer": f"No valid {label} observations are available."}
    ascending = not highest
    winner = table.sort_values(
        ["conversion_rate", "observations"], ascending=[ascending, False]
    ).iloc[0]
    superlative = "highest" if highest else "lowest"
    return {
        "answer": (
            f"**{winner['segment_value']}** has the {superlative} observed conversion rate: "
            f"**{winner['conversion_rate']:.2%}** "
            f"({int(winner['conversions'])}/{int(winner['observations'])} observations)."
        ),
        "table": table.sort_values("conversion_rate", ascending=ascending).reset_index(drop=True),
    }


def _revenue_answer(df: pd.DataFrame, column: str, label: str, highest: bool = True) -> dict:
    if "revenue" not in df.columns:
        return {"answer": "This dataset has no revenue column."}
    if column not in df.columns:
        return {"answer": f"This dataset has no '{column}' column."}
    table = _segment_table(df, column)
    if table.empty or "total_revenue" not in table.columns or not table["total_revenue"].notna().any():
        return {"answer": "No valid revenue observations available."}
    ascending = not highest
    superlative = "highest" if highest else "lowest"
    winner = table.sort_values("total_revenue", ascending=ascending).iloc[0]
    return {
        "answer": (
            f"**{winner['segment_value']}** has the {superlative} total revenue: "
            f"**${winner['total_revenue']:,.2f}** "
            f"(avg ${winner['revenue_per_observation']:,.2f} per observation)."
        ),
        "table": table.sort_values("total_revenue", ascending=ascending).reset_index(drop=True),
    }


def _experiment_answer(df: pd.DataFrame) -> dict:
    if not {"group", "converted"}.issubset(df.columns):
        return {"answer": "Need 'group' and 'converted' columns for experiment comparison."}
    control = df[df["group"] == "Control"]
    treatment = df[df["group"] == "Treatment"]
    result = run_two_proportion_test(control, treatment)
    if pd.isna(result["p_value"]):
        return {"answer": result["interpretation"], "warnings": result["warnings"]}
    ctrl_rate = calculate_conversion_rate(_binary_valid(control.get("converted", pd.Series(dtype=float))))
    trt_rate = calculate_conversion_rate(_binary_valid(treatment.get("converted", pd.Series(dtype=float))))
    uplift = trt_rate - ctrl_rate
    verdict = (
        "finds evidence of a difference"
        if result["p_value"] < result["alpha"]
        else "does not find sufficient evidence of a difference"
    )
    return {
        "answer": (
            f"Treatment conversion is **{abs(uplift):.2%} {'higher' if uplift >= 0 else 'lower'}** "
            f"than Control. The test **{verdict}** at α={result['alpha']:.2f}.\n\n"
            f"- Control: {ctrl_rate:.2%} ({result['control_successes']}/{result['control_total']})\n"
            f"- Treatment: {trt_rate:.2%} ({result['treatment_successes']}/{result['treatment_total']})\n"
            f"- Method: {result['method']}; p = {result['p_value']:.4g}"
        ),
        "table": pd.DataFrame([result]),
        "warnings": result["warnings"],
    }


def _overall_conversion_answer(df: pd.DataFrame) -> dict:
    if "converted" not in df.columns:
        return {"answer": "No 'converted' column found."}
    vals = _binary_valid(df["converted"])
    if vals.empty:
        return {"answer": "No valid binary conversion observations."}
    rate = float(vals.mean())
    return {"answer": f"The overall conversion rate across the dataset is **{rate:.2%}** ({int(vals.sum())} / {len(vals)} rows)."}


def _group_conversion_answer(df: pd.DataFrame, group_name: str) -> dict:
    if "group" not in df.columns or "converted" not in df.columns:
        return {"answer": "Need 'group' and 'converted' columns."}
    subset = df[df["group"].astype(str).str.casefold() == group_name.casefold()]
    if subset.empty:
        return {"answer": f"No rows found for group '{group_name}'."}
    vals = _binary_valid(subset["converted"])
    rate = float(vals.mean()) if not vals.empty else 0.0
    return {"answer": f"**{group_name}** conversion rate: **{rate:.2%}** ({int(vals.sum())} / {len(vals)} observations)."}


def _bounce_rate_answer(df: pd.DataFrame, column: str | None = None) -> dict:
    if "bounce" not in df.columns:
        return {"answer": "No 'bounce' column in this dataset."}
    if column is not None:
        return _conversion_answer(df.assign(converted=df["bounce"]), column, column.replace("_", " "), highest=False)
    vals = _binary_valid(df["bounce"])
    if vals.empty:
        return {"answer": "No valid binary bounce observations."}
    rate = float(vals.mean())
    return {"answer": f"Overall bounce rate: **{rate:.2%}** ({int(vals.sum())} / {len(vals)} rows)."}


def _revenue_per_user_answer(df: pd.DataFrame) -> dict:
    if "revenue" not in df.columns:
        return {"answer": "No 'revenue' column in this dataset."}
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    total = float(rev.sum(min_count=1))
    n_users = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)
    rpu = total / n_users if n_users > 0 else 0.0
    return {"answer": f"Average revenue per user: **${rpu:,.2f}** (total ${total:,.2f} across {n_users:,} users)."}


def _total_revenue_answer(df: pd.DataFrame) -> dict:
    if "revenue" not in df.columns:
        return {"answer": "No 'revenue' column in this dataset."}
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    total = float(rev.sum(min_count=1))
    return {"answer": f"Total revenue in the filtered dataset: **${total:,.2f}**."}


def _revenue_per_conversion_answer(df: pd.DataFrame) -> dict:
    if "revenue" not in df.columns or "converted" not in df.columns:
        return {"answer": "Need 'revenue' and 'converted' columns."}
    converted_rows = df[_binary_valid(df["converted"]) == 1]
    rev = pd.to_numeric(converted_rows["revenue"], errors="coerce").dropna()
    if rev.empty:
        return {"answer": "No revenue observed for converted users."}
    rpc = float(rev.mean())
    return {"answer": f"Average revenue per conversion: **${rpc:,.2f}** (across {len(rev):,} converted rows)."}


def _session_duration_answer(df: pd.DataFrame, column: str | None = None, highest: bool = True) -> dict:
    if "session_duration" not in df.columns:
        return {"answer": "No 'session_duration' column in this dataset."}
    if column is not None and column in df.columns:
        df2 = df.copy()
        df2["_sd"] = pd.to_numeric(df2["session_duration"], errors="coerce")
        tbl = df2.groupby(column)["_sd"].mean().reset_index()
        tbl.columns = [column, "avg_session_duration"]
        tbl = tbl.sort_values("avg_session_duration", ascending=not highest)
        winner = tbl.iloc[0]
        superlative = "longest" if highest else "shortest"
        return {
            "answer": f"**{winner[column]}** has the {superlative} average session duration: **{winner['avg_session_duration']:.1f}s**.",
            "table": tbl.reset_index(drop=True),
        }
    sd = pd.to_numeric(df["session_duration"], errors="coerce").dropna()
    if sd.empty:
        return {"answer": "No valid session_duration observations."}
    return {"answer": f"Average session duration: **{sd.mean():.1f}s** (min {sd.min():.0f}s, max {sd.max():.0f}s)."}


def _pages_viewed_answer(df: pd.DataFrame) -> dict:
    if "pages_viewed" not in df.columns:
        return {"answer": "No 'pages_viewed' column in this dataset."}
    pv = pd.to_numeric(df["pages_viewed"], errors="coerce").dropna()
    if pv.empty:
        return {"answer": "No valid pages_viewed observations."}
    return {"answer": f"Average pages viewed per session: **{pv.mean():.2f}** (median {pv.median():.1f})."}


def _weekday_conversion_answer(df: pd.DataFrame) -> dict:
    if "date" not in df.columns or "converted" not in df.columns:
        return {"answer": "Need 'date' and 'converted' columns for weekday analysis."}
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce", format="mixed")
    df2["converted"] = pd.to_numeric(df2["converted"], errors="coerce")
    df2 = df2[df2["date"].notna() & df2["converted"].isin([0, 1])]
    if df2.empty:
        return {"answer": "No valid date + conversion observations."}
    df2["weekday"] = df2["date"].dt.day_name()
    tbl = df2.groupby("weekday")["converted"].agg(["mean", "count"]).reset_index()
    tbl.columns = ["Weekday", "conversion_rate", "observations"]
    tbl = tbl.sort_values("conversion_rate", ascending=False).reset_index(drop=True)
    winner = tbl.iloc[0]
    return {
        "answer": (
            f"**{winner['Weekday']}** has the highest observed conversion rate: "
            f"**{winner['conversion_rate']:.2%}** ({int(winner['observations'])} observations)."
        ),
        "table": tbl,
    }


def _weekly_conversion_answer(df: pd.DataFrame) -> dict:
    if "date" not in df.columns or "converted" not in df.columns:
        return {"answer": "Need 'date' and 'converted' columns for weekly analysis."}
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce", format="mixed")
    df2["converted"] = pd.to_numeric(df2["converted"], errors="coerce")
    df2 = df2[df2["date"].notna() & df2["converted"].isin([0, 1])]
    if df2.empty:
        return {"answer": "No valid date + conversion observations."}
    df2["week"] = df2["date"].dt.to_period("W").astype(str)
    tbl = df2.groupby("week")["converted"].agg(["mean", "count"]).reset_index()
    tbl.columns = ["Week", "conversion_rate", "observations"]
    tbl = tbl.sort_values("conversion_rate", ascending=False).reset_index(drop=True)
    winner = tbl.iloc[0]
    return {
        "answer": (
            f"Week **{winner['Week']}** had the highest observed conversion rate: "
            f"**{winner['conversion_rate']:.2%}** ({int(winner['observations'])} observations)."
        ),
        "table": tbl,
    }


def _trend_answer(df: pd.DataFrame) -> dict:
    if "date" not in df.columns or "converted" not in df.columns:
        return {"answer": "Need 'date' and 'converted' columns for trend analysis."}
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce", format="mixed")
    df2["converted"] = pd.to_numeric(df2["converted"], errors="coerce")
    df2 = df2[df2["date"].notna() & df2["converted"].isin([0, 1])]
    if df2.empty:
        return {"answer": "No valid date + conversion observations for trend."}
    df2["date_day"] = df2["date"].dt.date
    tbl = df2.groupby("date_day")["converted"].agg(["mean", "count"]).reset_index()
    tbl.columns = ["Date", "conversion_rate", "observations"]
    first_rate = float(tbl.iloc[0]["conversion_rate"])
    last_rate = float(tbl.iloc[-1]["conversion_rate"])
    delta = last_rate - first_rate
    direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "stayed flat")
    return {
        "answer": (
            f"Conversion rate {direction} from **{first_rate:.2%}** on {tbl.iloc[0]['Date']} "
            f"to **{last_rate:.2%}** on {tbl.iloc[-1]['Date']} "
            f"({delta:+.2%} over {len(tbl)} days)."
        ),
        "table": tbl,
    }


def _topn_answer(df: pd.DataFrame, column: str, metric: str, n: int = 3, highest: bool = True) -> dict:
    if column not in df.columns:
        return {"answer": f"No '{column}' column in this dataset."}
    if metric == "conversion" and "converted" in df.columns:
        table = _segment_table(df, column)
        if table.empty:
            return {"answer": "No valid observations."}
        table = table.sort_values("conversion_rate", ascending=not highest).head(n)
        direction = "Top" if highest else "Bottom"
        rows_text = "\n".join(
            f"- {row['segment_value']}: {row['conversion_rate']:.2%}"
            for _, row in table.iterrows()
        )
        return {
            "answer": f"**{direction} {n} {column} by conversion rate:**\n{rows_text}",
            "table": table.reset_index(drop=True),
        }
    if metric == "revenue" and "revenue" in df.columns:
        table = _segment_table(df, column)
        if table.empty or "total_revenue" not in table.columns:
            return {"answer": "No valid revenue observations."}
        table = table.sort_values("total_revenue", ascending=not highest).head(n)
        direction = "Top" if highest else "Bottom"
        rows_text = "\n".join(
            f"- {row['segment_value']}: ${row['total_revenue']:,.2f}"
            for _, row in table.iterrows()
        )
        return {
            "answer": f"**{direction} {n} {column} by total revenue:**\n{rows_text}",
            "table": table.reset_index(drop=True),
        }
    return {"answer": f"Cannot compute top-N for metric '{metric}'. Try 'conversion' or 'revenue'."}


def _user_count_answer(df: pd.DataFrame) -> dict:
    n_users = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)
    ctrl = len(df[df["group"] == "Control"]) if "group" in df.columns else None
    trt = len(df[df["group"] == "Treatment"]) if "group" in df.columns else None
    detail = ""
    if ctrl is not None:
        detail = f" (Control: {ctrl:,}, Treatment: {trt:,})"
    return {"answer": f"There are **{n_users:,}** users in the filtered dataset{detail}."}


def _top_findings(df: pd.DataFrame) -> list[str]:
    findings = []
    for col, lbl in (("device", "device"), ("traffic_source", "traffic source")):
        if col in df.columns:
            findings.append(_conversion_answer(df, col, lbl)["answer"])
    if {"group", "converted"}.issubset(df.columns):
        findings.append(_experiment_answer(df)["answer"])
    if "revenue" in df.columns:
        for col in ("device", "traffic_source", "country", "gender"):
            if col in df.columns:
                findings.append(_revenue_answer(df, col, col.replace("_", " "), highest=True)["answer"])
                break
    return findings[:4]


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

def _extract_number(text: str, default: int = 3) -> int:
    """Extract the first integer found in text, or return default."""
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else default


def _detect_column(text: str, candidates: list[str]) -> str | None:
    for col in candidates:
        if col.replace("_", " ") in text or col in text:
            return col
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SEGMENT_COLS = ["device", "traffic_source", "country", "gender", "page_type"]


def answer_question(df: pd.DataFrame, question: str) -> dict:
    """Answer supported business questions using only values in the supplied DataFrame.

    Returns a dict with keys:
    - ``answer`` (str): markdown-formatted answer
    - ``table`` (pd.DataFrame, optional): supporting data table
    - ``warnings`` (list[str], optional): advisory messages
    """
    if not question or not question.strip():
        return {
            "answer": (
                "Enter a question about this dataset. Examples:\n"
                "- 'Which device has the highest conversion rate?'\n"
                "- 'Did Treatment improve conversion?'\n"
                "- 'Top 3 countries by conversion'\n"
                "- 'What is the average revenue per user?'\n"
                "- 'Which day of the week converts best?'"
            )
        }

    text = question.casefold().strip()

    # ---- Top / key findings ------------------------------------------------
    if any(t in text for t in ("key finding", "top finding", "important finding",
                                "3 most", "three most", "main finding")):
        findings = _top_findings(df)
        return {
            "answer": "\n".join(f"{i}. {f}" for i, f in enumerate(findings, 1))
            if findings else "No supported findings can be calculated from the available columns.",
        }

    # ---- User / sample count -----------------------------------------------
    if any(t in text for t in ("how many user", "number of user", "sample size", "how many participant")):
        return _user_count_answer(df)

    # ---- Overall conversion ------------------------------------------------
    if ("overall" in text or "total" in text) and "conversion" in text and "rate" in text:
        return _overall_conversion_answer(df)

    if "what percentage" in text and "convert" in text:
        return _overall_conversion_answer(df)

    # ---- Group-level conversion --------------------------------------------
    for group_name in ("control", "treatment"):
        if group_name in text and ("conversion" in text or "convert" in text or "rate" in text):
            return _group_conversion_answer(df, group_name.capitalize())

    # ---- Experiment comparison (Treatment vs Control) ----------------------
    if "treatment" in text and any(t in text for t in ("improve", "better", "effect", "did", "help", "convert")):
        return _experiment_answer(df)
    if "compare" in text and ("control" in text or "treatment" in text):
        return _experiment_answer(df)

    # ---- Top-N queries -----------------------------------------------------
    top_match = re.search(r"\b(top|bottom)\s*(\d+)\s+(\w+)\s+by\s+(\w+)", text)
    if top_match:
        direction_str, n_str, col_str, metric_str = top_match.groups()
        n = int(n_str)
        highest = direction_str == "top"
        # Map plural forms and aliases to actual column names
        col_aliases = {
            "countries": "country",
            "devices": "device",
            "sources": "traffic_source",
            "genders": "gender",
            "pages": "page_type",
        }
        col_str_norm = col_aliases.get(col_str, col_str)
        col = _detect_column(col_str_norm, SEGMENT_COLS) or col_str_norm
        metric = "revenue" if "revenue" in metric_str else "conversion"
        return _topn_answer(df, col, metric, n=n, highest=highest)

    # ---- Bounce rate -------------------------------------------------------
    if "bounce" in text:
        if any(t in text for t in ("lowest", "best", "which device", "which source")):
            col = _detect_column(text, SEGMENT_COLS) or "device"
            return _bounce_rate_answer(df, column=col)
        return _bounce_rate_answer(df)

    # ---- Revenue per user --------------------------------------------------
    if "revenue per user" in text or "arpu" in text or "average revenue per" in text:
        return _revenue_per_user_answer(df)

    # ---- Revenue per conversion --------------------------------------------
    if "revenue per conversion" in text or "revenue per converted" in text:
        return _revenue_per_conversion_answer(df)

    # ---- Total revenue -----------------------------------------------------
    if "total revenue" in text:
        return _total_revenue_answer(df)

    # ---- Highest / lowest revenue by segment -------------------------------
    if "revenue" in text:
        col = _detect_column(text, SEGMENT_COLS)
        highest = not any(t in text for t in ("lowest", "least", "worst", "bottom"))
        if col:
            return _revenue_answer(df, col, col.replace("_", " "), highest=highest)
        # Fallback: first available segment col
        col = next((c for c in SEGMENT_COLS if c in df.columns), None)
        if col:
            return _revenue_answer(df, col, col.replace("_", " "), highest=highest)

    # ---- Session duration --------------------------------------------------
    if "session" in text and ("duration" in text or "long" in text or "short" in text):
        col = _detect_column(text, SEGMENT_COLS)
        highest = not any(t in text for t in ("shortest", "lowest", "worst"))
        return _session_duration_answer(df, column=col, highest=highest)

    # ---- Pages viewed -------------------------------------------------------
    if "page" in text and ("view" in text or "average" in text):
        return _pages_viewed_answer(df)

    # ---- Weekday analysis --------------------------------------------------
    if ("day" in text or "weekday" in text) and any(t in text for t in ("best", "highest", "convert", "which")):
        return _weekday_conversion_answer(df)

    # ---- Weekly analysis ---------------------------------------------------
    if "week" in text and any(t in text for t in ("best", "highest", "convert", "which")):
        return _weekly_conversion_answer(df)

    # ---- Trend over time ---------------------------------------------------
    if "trend" in text or ("over time" in text) or ("time series" in text):
        return _trend_answer(df)

    # ---- Segment conversion (device, source, country, gender, page_type) ---
    for col, labels in (
        ("device", ("device", "devices", "mobile", "tablet", "web", "desktop")),
        ("traffic_source", ("traffic source", "source", "organic", "paid", "social", "referral", "email")),
        ("country", ("country", "countries", "region", "geography")),
        ("gender", ("gender", "male", "female")),
        ("page_type", ("page type", "page", "landing", "pricing", "checkout")),
    ):
        if any(lbl in text for lbl in labels):
            highest = not any(t in text for t in ("lowest", "worst", "least", "bottom"))
            if any(t in text for t in ("conversion", "convert", "best", "highest", "lowest", "rate")):
                return _conversion_answer(df, col, col.replace("_", " "), highest=highest)

    # ---- Fallback ----------------------------------------------------------
    return {
        "answer": (
            "I can answer questions about:\n"
            "- Conversion rates (device, traffic source, country, gender, Control/Treatment)\n"
            "- Revenue (per user, per conversion, total, by segment)\n"
            "- Bounce rate and session duration\n"
            "- Weekday / weekly trends\n"
            "- Top-N queries (e.g., 'Top 3 countries by conversion')\n"
            "- Key experiment findings\n\n"
            "Please rephrase your question using one of those topics."
        )
    }
