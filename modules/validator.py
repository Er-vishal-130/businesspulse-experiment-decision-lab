from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["group", "converted"]
OPTIONAL_COLUMNS = [
    "user_id",
    "session_id",
    "experiment_id",
    "revenue",
    "age",
    "gender",
    "country",
    "device",
    "traffic_source",
    "session_duration",
    "pages_viewed",
    "bounce",
    "error_flag",
    "date",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    cleaned = df.copy()
    if "group" in cleaned.columns:
        cleaned["group"] = cleaned["group"].astype(str).str.strip().str.title()
    if "converted" in cleaned.columns:
        conversion_values = cleaned["converted"].astype(str).str.strip().str.lower()
        cleaned["converted"] = conversion_values.map(
            {"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0}
        )
    for col in [
        "user_id",
        "age",
        "pages_viewed",
        "session_duration",
        "bounce",
        "error_flag",
        "revenue",
    ]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce", format="mixed")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        if col in cleaned.columns:
            cleaned = cleaned.dropna(subset=[col])
    if "group" in cleaned.columns:
        cleaned = cleaned[cleaned["group"].isin(["Control", "Treatment"])].reset_index(drop=True)
    if "converted" in cleaned.columns:
        cleaned = cleaned[cleaned["converted"].isin([0, 1])].reset_index(drop=True)
    return cleaned


def validate_dataframe(df: pd.DataFrame) -> dict:
    issues: list[str] = []
    if df is None or df.empty:
        return {
            "status": "Invalid",
            "issues": ["Dataset is empty."],
            "missing_values": 0,
            "duplicate_rows": 0,
            "duplicate_user_ids": 0,
            "rows": 0,
            "columns": 0,
        }

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_user_ids = int(df["user_id"].duplicated().sum()) if "user_id" in df.columns else 0

    if missing_values:
        issues.append(f"Missing values detected: {missing_values}")
    if duplicate_rows:
        issues.append(f"Duplicate rows detected: {duplicate_rows}")
    if duplicate_user_ids:
        issues.append(f"Duplicate user IDs detected: {duplicate_user_ids}")
    if "converted" in df.columns:
        converted = pd.to_numeric(df["converted"], errors="coerce")
        weird_converted = int((converted.notna() & ~converted.isin([0, 1])).sum())
        weird_converted += int((df["converted"].notna() & converted.isna()).sum())
        if weird_converted:
            issues.append(f"Invalid conversion values detected: {weird_converted}")
    if "group" in df.columns:
        invalid_groups = set(df["group"].dropna().astype(str).str.strip().str.title()) - {"Control", "Treatment"}
        if invalid_groups:
            issues.append(f"Invalid group names: {sorted(invalid_groups)}")
    if "revenue" in df.columns:
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        invalid_revenue = int((df["revenue"].notna() & revenue.isna()).sum())
        negative_revenue = int((revenue < 0).sum())
        if invalid_revenue:
            issues.append(f"Invalid revenue values detected: {invalid_revenue}")
        if negative_revenue:
            issues.append(f"Negative revenue values: {negative_revenue}")
    if "date" in df.columns:
        invalid_dates = int(
            (df["date"].notna() & pd.to_datetime(df["date"], errors="coerce", format="mixed").isna()).sum()
        )
        if invalid_dates:
            issues.append(f"Invalid date values detected: {invalid_dates}")

    invalid_numeric = 0
    for column in ["age", "session_duration", "pages_viewed", "bounce", "error_flag"]:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid_numeric += int((df[column].notna() & numeric.isna()).sum())
    if invalid_numeric:
        issues.append(f"Invalid numeric values detected: {invalid_numeric}")

    if issues:
        status = "Warning" if len(issues) <= 3 else "Invalid"
    else:
        status = "Valid"

    return {
        "status": status,
        "issues": issues,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_user_ids": duplicate_user_ids,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }
