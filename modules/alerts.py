from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd


DEFAULT_SEVERITY_THRESHOLDS = {
    "medium": 0.02,
    "high": 0.05,
    "critical": 0.10,
}
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
ALERT_STATUSES = ("New", "Investigating", "Resolved")


def _valid_thresholds(thresholds: Mapping[str, float]) -> tuple[float, float, float]:
    values = tuple(float(thresholds[key]) for key in ("medium", "high", "critical"))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Severity thresholds must be finite and non-negative.")
    if not values[0] < values[1] < values[2]:
        raise ValueError("Severity thresholds must increase from medium to critical.")
    return values


def _severity(absolute_change: float, thresholds: Mapping[str, float]) -> tuple[str, str]:
    medium, high, critical = _valid_thresholds(thresholds)
    magnitude_pp = abs(absolute_change) * 100
    if magnitude_pp >= critical * 100:
        return "Critical", f"|change| {magnitude_pp:.2f} pp >= configured {critical * 100:.2f} pp"
    if magnitude_pp >= high * 100:
        return "High", f"|change| {magnitude_pp:.2f} pp >= configured {high * 100:.2f} pp"
    if magnitude_pp >= medium * 100:
        return "Medium", f"|change| {magnitude_pp:.2f} pp >= configured {medium * 100:.2f} pp"
    return "Low", f"|change| {magnitude_pp:.2f} pp < configured {medium * 100:.2f} pp"


def build_alerts(
    detection: Mapping[str, Any],
    *,
    source_key: str,
    metric_name: str = "Conversion rate",
    thresholds: Mapping[str, float] = DEFAULT_SEVERITY_THRESHOLDS,
    detected_at: str | None = None,
) -> list[dict[str, Any]]:
    """Create alerts only from existing detector-supported anomaly rows."""
    rows = detection.get("rows")
    if not detection.get("inferential_valid", False) or not isinstance(rows, pd.DataFrame) or rows.empty:
        return []
    if not {"segment", "segment_column", "absolute_lift", "is_anomaly"}.issubset(rows.columns):
        return []
    _valid_thresholds(thresholds)
    timestamp = detected_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alerts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows.to_dict(orient="records"):
        anomaly_flag = row.get("is_anomaly", False)
        if pd.isna(anomaly_flag) or not bool(anomaly_flag):
            continue
        try:
            observed = float(row["observed_rate"])
            expected = float(row["expected_rate"])
            absolute_change = float(row["absolute_lift"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (observed, expected, absolute_change)):
            continue
        segment_column = str(row["segment_column"])
        segment = str(row["segment"])
        identity = json.dumps(
            [source_key, metric_name, segment_column, segment],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        alert_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        if alert_id in seen_ids:
            continue
        seen_ids.add(alert_id)
        severity, rationale = _severity(absolute_change, thresholds)
        relative = row.get("relative_lift_pct")
        try:
            relative_change = float(relative) if relative is not None else None
        except (TypeError, ValueError):
            relative_change = None
        if relative_change is not None and not math.isfinite(relative_change):
            relative_change = None
        alerts.append(
            {
                "alert_id": alert_id,
                "metric": metric_name,
                "detected_at": timestamp,
                "observed_value": observed,
                "expected_value": expected,
                "absolute_change": absolute_change,
                "percentage_change": relative_change,
                "segment_column": segment_column,
                "segment": segment,
                "severity": severity,
                "severity_rationale": rationale,
                "status": "New",
                "statistical_status": "Adjusted-significant exploratory finding",
                "evidence": row,
            }
        )
    return alerts


def filter_and_sort_alerts(
    alerts: Sequence[Mapping[str, Any]],
    *,
    severities: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    segments: Sequence[str] | None = None,
    sort_by: str = "Severity",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Filter alert fields and apply stable severity, date, or magnitude sorting."""
    valid_sort_keys = {"Severity", "Date", "Magnitude"}
    if sort_by not in valid_sort_keys:
        raise ValueError(f"sort_by must be one of {sorted(valid_sort_keys)}.")
    selected = []
    for alert in alerts:
        if severities and alert.get("severity") not in severities:
            continue
        if statuses and alert.get("status") not in statuses:
            continue
        if metrics and alert.get("metric") not in metrics:
            continue
        segment_label = (
            f"{alert.get('segment_column')} = {alert.get('segment')}"
        )
        if segments and alert.get("segment") not in segments and segment_label not in segments:
            continue
        selected.append(dict(alert))
    if sort_by == "Severity":
        key = lambda alert: SEVERITY_ORDER.get(str(alert.get("severity")), len(SEVERITY_ORDER))
        selected.sort(key=key, reverse=not descending)
    elif sort_by == "Date":
        selected.sort(key=lambda alert: str(alert.get("detected_at") or ""), reverse=descending)
    else:
        selected.sort(
            key=lambda alert: abs(float(alert.get("absolute_change", 0.0) or 0.0)),
            reverse=descending,
        )
    return selected
