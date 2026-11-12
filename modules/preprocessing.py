from __future__ import annotations

import pandas as pd


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "group" in normalized.columns:
        normalized["group"] = normalized["group"].astype(str).str.strip().str.title()
    if "converted" in normalized.columns:
        normalized["converted"] = pd.to_numeric(normalized["converted"], errors="coerce")
    if "revenue" in normalized.columns:
        normalized["revenue"] = pd.to_numeric(normalized["revenue"], errors="coerce")
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", format="mixed")
    return normalized


def filter_dataset(df: pd.DataFrame, **filters) -> pd.DataFrame:
    filtered = df.copy()
    for column in ("group", "device", "country", "traffic_source"):
        selected_values = filters.get(column)
        if selected_values is None or column not in filtered.columns:
            continue
        selected = {str(value) for value in selected_values}
        available = set(df[column].dropna().astype(str).unique())
        if selected != available:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
    if "date_min" in filters and filters["date_min"] is not None and "date" in filtered:
        filtered = filtered[filtered["date"] >= pd.to_datetime(filters["date_min"])]
    if "date_max" in filters and filters["date_max"] is not None and "date" in filtered:
        end_date = pd.to_datetime(filters["date_max"]).date()
        filtered = filtered[filtered["date"].dt.date <= end_date]
    if "min_age" in filters and filters["min_age"] is not None and "age" in filtered:
        filtered = filtered[pd.to_numeric(filtered["age"], errors="coerce") >= filters["min_age"]]
    if "max_age" in filters and filters["max_age"] is not None and "age" in filtered:
        filtered = filtered[pd.to_numeric(filtered["age"], errors="coerce") <= filters["max_age"]]
    return filtered.reset_index(drop=True)
