from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_FILE = DATA_DIR / "sample_ab_test.csv"
KAGGLE_FILE = DATA_DIR / "ab_data.csv"


def load_kaggle_dataset() -> pd.DataFrame:
    """Load the bundled Kaggle A/B dataset into the platform's column conventions."""
    if not KAGGLE_FILE.is_file():
        raise FileNotFoundError(f"Kaggle dataset file not found: {KAGGLE_FILE}")

    df = pd.read_csv(KAGGLE_FILE)
    required_columns = {"user_id", "timestamp", "group", "landing_page", "converted"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kaggle dataset is missing required columns: {missing}")

    return df.rename(
        columns={
            "timestamp": "date",
            "landing_page": "page_type",
        }
    )


def generate_sample_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(int(rows), 100)
    group = rng.choice(["Control", "Treatment"], size=n, p=[0.5, 0.5])
    control_probability = 0.12
    treatment_probability = 0.138
    converted = np.where(
        group == "Treatment",
        rng.random(n) < treatment_probability,
        rng.random(n) < control_probability,
    )
    revenue = np.where(converted, rng.integers(100, 900, size=n), 0)
    age = rng.integers(18, 70, size=n)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
    country = rng.choice(
        ["India", "USA", "UK", "Germany", "Canada", "Australia"],
        size=n,
        p=[0.35, 0.25, 0.16, 0.12, 0.07, 0.05],
    )
    device = rng.choice(["Mobile", "Web", "Tablet"], size=n, p=[0.52, 0.38, 0.10])
    traffic_source = rng.choice(
        ["Organic", "Paid Search", "Social", "Referral", "Email"],
        size=n,
        p=[0.30, 0.22, 0.18, 0.15, 0.15],
    )
    session_duration = rng.integers(15, 900, size=n)
    pages_viewed = rng.integers(1, 12, size=n)
    bounce = rng.choice([0, 1], size=n, p=[0.68, 0.32])
    error_flag = rng.choice([0, 1], size=n, p=[0.88, 0.12])
    date = pd.date_range(start="2024-01-01", periods=n, freq="h").tolist()

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "experiment_id": "exp_001",
            "group": group,
            "converted": converted.astype(int),
            "revenue": revenue.astype(float),
            "age": age,
            "gender": gender,
            "country": country,
            "device": device,
            "traffic_source": traffic_source,
            "session_duration": session_duration.astype(float),
            "pages_viewed": pages_viewed.astype(int),
            "bounce": bounce.astype(int),
            "error_flag": error_flag.astype(int),
            "date": date,
        }
    )
    return df


def ensure_sample_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_FILE.exists():
        df = generate_sample_dataset()
        df.to_csv(SAMPLE_FILE, index=False)
    return pd.read_csv(SAMPLE_FILE)


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def build_upload_metadata(
    filename: str,
    file_bytes: bytes,
    df: pd.DataFrame,
    uploaded_at: str | None = None,
) -> dict:
    """Return session-safe upload metadata without retaining uploaded contents."""
    return {
        "filename": filename,
        "uploaded_at": uploaded_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count
        },
        "file_size_bytes": len(file_bytes),
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
    }
