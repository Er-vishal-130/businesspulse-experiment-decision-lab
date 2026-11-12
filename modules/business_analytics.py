from __future__ import annotations

import pandas as pd


def summarize_time_trend(df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    """Aggregate observed conversion and revenue by date and experiment group."""
    required = {"date", "group", "converted"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    frame = df.loc[:, ["date", "group", "converted"] + (["revenue"] if "revenue" in df else [])].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    frame["converted"] = pd.to_numeric(frame["converted"], errors="coerce")
    frame = frame.dropna(subset=["date", "group", "converted"])
    frame = frame[frame["converted"].isin([0, 1])]
    if frame.empty:
        return pd.DataFrame()

    frame["date"] = frame["date"].dt.to_period(frequency).dt.start_time
    aggregation = {
        "observations": ("converted", "size"),
        "conversions": ("converted", "sum"),
        "conversion_rate": ("converted", "mean"),
    }
    if "revenue" in frame:
        frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
        aggregation["revenue"] = ("revenue", "sum")
    return (
        frame.groupby(["date", "group"], as_index=False, observed=True)
        .agg(**aggregation)
        .sort_values(["date", "group"])
        .reset_index(drop=True)
    )


def categorical_segment_columns(df: pd.DataFrame, max_categories: int = 50) -> list[str]:
    excluded = {"group", "converted", "user_id", "session_id", "experiment_id", "date"}
    columns = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        if 1 < df[column].nunique(dropna=True) <= max_categories:
            columns.append(column)
    return columns
