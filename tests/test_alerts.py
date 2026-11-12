import pandas as pd
import pytest

from modules.alerts import (
    ALERT_STATUSES,
    build_alerts,
    filter_and_sort_alerts,
)
from modules.detective import detect_business_anomaly


def make_alert_fixture() -> pd.DataFrame:
    rows = []
    for index in range(500):
        rows.append(
            {
                "user_id": f"control-{index}",
                "group": "Control",
                "device": "Control",
                "converted": int(index < 50),
            }
        )
    for segment, conversions in (("Moderate", 150), ("Strong", 300)):
        for index in range(500):
            rows.append(
                {
                    "user_id": f"{segment}-{index}",
                    "group": "Treatment",
                    "device": segment,
                    "converted": int(index < conversions),
                }
            )
    return pd.DataFrame(rows)


def thresholds() -> dict[str, float]:
    return {"medium": 0.02, "high": 0.15, "critical": 0.25}


def test_alerts_reuse_existing_significant_detector_rows_and_assign_severity():
    detection = detect_business_anomaly(
        make_alert_fixture(), segment_by="device", min_observations=30
    )

    alerts = build_alerts(
        detection,
        source_key="sample",
        thresholds=thresholds(),
        detected_at="2026-10-16T12:00:00+00:00",
    )

    assert {alert["segment"] for alert in alerts} == {"Moderate", "Strong"}
    by_segment = {alert["segment"]: alert for alert in alerts}
    assert by_segment["Moderate"]["severity"] == "High"
    assert "configured 15.00 pp" in by_segment["Moderate"]["severity_rationale"]
    assert by_segment["Strong"]["severity"] == "Critical"
    assert by_segment["Strong"]["observed_value"] == pytest.approx(0.6)
    assert by_segment["Strong"]["expected_value"] == pytest.approx(0.1)
    assert by_segment["Strong"]["absolute_change"] == pytest.approx(0.5)
    assert by_segment["Strong"]["percentage_change"] == pytest.approx(500)
    assert by_segment["Strong"]["status"] == "New"
    assert by_segment["Strong"]["detected_at"] == "2026-10-16T12:00:00+00:00"

    medium_rule_alerts = build_alerts(
        detection,
        source_key="sample",
        thresholds={"medium": 0.01, "high": 0.30, "critical": 0.40},
        detected_at="fixed",
    )
    assert next(
        alert for alert in medium_rule_alerts if alert["segment"] == "Moderate"
    )["severity"] == "Medium"
    low_rule_alerts = build_alerts(
        detection,
        source_key="sample",
        thresholds={"medium": 0.21, "high": 0.30, "critical": 0.40},
        detected_at="fixed",
    )
    assert next(
        alert for alert in low_rule_alerts if alert["segment"] == "Moderate"
    )["severity"] == "Low"


def test_zero_baseline_has_no_percentage_change_and_non_significant_rows_are_not_alerts():
    rows = []
    for index in range(5000):
        rows.append(
            {
                "user_id": f"c{index}",
                "group": "Control",
                "device": "Control",
                "converted": 0,
            }
        )
    for index in range(5000):
        rows.append(
            {
                "user_id": f"t{index}",
                "group": "Treatment",
                "device": "Mobile",
                "converted": int(index < 100),
            }
        )
    result = detect_business_anomaly(
        pd.DataFrame(rows), segment_by="device", min_observations=30
    )
    alert = build_alerts(
        result,
        source_key="upload:zero-baseline",
        thresholds=thresholds(),
        detected_at="fixed",
    )[0]

    assert alert["expected_value"] == 0
    assert alert["percentage_change"] is None
    assert alert["evidence"]["test_method"] == "Two-proportion score test"

    normal_rows = []
    for group, segment in (
        ("Control", "Control"),
        ("Treatment", "Moderate"),
        ("Treatment", "Strong"),
    ):
        for index in range(500):
            normal_rows.append(
                {
                    "user_id": f"{group}-{segment}-{index}",
                    "group": group,
                    "device": segment,
                    "converted": int(index < 50),
                }
            )
    normal_result = detect_business_anomaly(
        pd.DataFrame(normal_rows), segment_by="device", min_observations=30
    )
    assert build_alerts(normal_result, source_key="sample", thresholds=thresholds()) == []


def test_repeated_identifiers_missing_data_and_empty_results_do_not_create_alerts():
    frame = make_alert_fixture()
    frame.loc[frame.index[-1], "user_id"] = frame.loc[frame.index[-250], "user_id"]
    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    assert build_alerts(result, source_key="sample", thresholds=thresholds()) == []

    empty_result = detect_business_anomaly(
        pd.DataFrame(columns=["group", "device", "converted"]),
        segment_by="device",
    )
    assert build_alerts(empty_result, source_key="empty", thresholds=thresholds()) == []


def test_non_finite_detector_values_are_skipped_without_losing_valid_alerts():
    detection = detect_business_anomaly(
        make_alert_fixture(), segment_by="device", min_observations=30
    )
    detection["rows"].loc[
        detection["rows"]["segment"] == "Strong", "observed_rate"
    ] = float("inf")

    alerts = build_alerts(
        detection, source_key="sample", thresholds=thresholds(), detected_at="fixed"
    )

    assert [alert["segment"] for alert in alerts] == ["Moderate"]


def test_alerts_are_deduplicated_and_filter_and_sort_are_deterministic():
    detection = detect_business_anomaly(
        make_alert_fixture(), segment_by="device", min_observations=30
    )
    duplicated_detection = dict(detection)
    duplicated_detection["rows"] = pd.concat(
        [detection["rows"], detection["rows"].iloc[[0]]], ignore_index=True
    )
    alerts = build_alerts(
        duplicated_detection,
        source_key="sample",
        thresholds=thresholds(),
        detected_at="2026-10-16T12:00:00+00:00",
    )
    assert len(alerts) == 2

    high_alert = next(alert for alert in alerts if alert["severity"] == "High")
    high_alert["status"] = "Investigating"
    high_alert["detected_at"] = "2026-10-16T13:00:00+00:00"
    critical_alert = next(alert for alert in alerts if alert["severity"] == "Critical")
    critical_alert["detected_at"] = "2026-10-16T12:00:00+00:00"
    investigating_high = filter_and_sort_alerts(
        alerts,
        severities=["High"],
        statuses=["Investigating"],
        metrics=["Conversion rate"],
        sort_by="Magnitude",
    )
    assert len(investigating_high) == 1
    assert investigating_high[0]["segment"] == "Moderate"
    dimension_filtered = filter_and_sort_alerts(
        alerts,
        segments=["device = Strong"],
    )
    assert [alert["segment"] for alert in dimension_filtered] == ["Strong"]

    severity_sorted = filter_and_sort_alerts(alerts, sort_by="Severity")
    assert [alert["severity"] for alert in severity_sorted] == ["Critical", "High"]
    magnitude_sorted = filter_and_sort_alerts(alerts, sort_by="Magnitude")
    assert magnitude_sorted[0]["segment"] == "Strong"
    date_sorted = filter_and_sort_alerts(alerts, sort_by="Date")
    assert date_sorted[0]["detected_at"] == "2026-10-16T13:00:00+00:00"
    assert {alert["status"] for alert in date_sorted}.issubset(ALERT_STATUSES)


@pytest.mark.parametrize(
    "values",
    [
        {"medium": 0.05, "high": 0.05, "critical": 0.1},
        {"medium": -0.01, "high": 0.05, "critical": 0.1},
        {"medium": 0.02, "high": float("nan"), "critical": 0.1},
    ],
)
def test_invalid_severity_thresholds_are_rejected(values):
    detection = detect_business_anomaly(
        make_alert_fixture(), segment_by="device", min_observations=30
    )
    with pytest.raises(ValueError):
        build_alerts(detection, source_key="sample", thresholds=values)
