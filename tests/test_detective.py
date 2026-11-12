import pandas as pd
import pytest

from modules.detective import detect_business_anomaly, generate_anomaly_demo_dataset


def make_ab_fixture(
    control_successes: int = 10,
    control_n: int = 100,
    treatment_successes_by_segment: dict[str, int] | None = None,
    treatment_n: int = 100,
) -> pd.DataFrame:
    treatment_successes_by_segment = treatment_successes_by_segment or {
        "Mobile": 10,
        "Web": 10,
    }
    rows = []
    for index in range(control_n):
        rows.append(
            {
                "user_id": f"c-{index}",
                "group": "Control",
                "device": "All",
                "converted": int(index < control_successes),
            }
        )
    for segment, successes in treatment_successes_by_segment.items():
        for index in range(treatment_n):
            rows.append(
                {
                    "user_id": f"t-{segment}-{index}",
                    "group": "Treatment",
                    "device": segment,
                    "converted": int(index < successes),
                }
            )
    return pd.DataFrame(rows)


def test_normal_data_reports_no_adjusted_alert_and_exact_rates():
    frame = make_ab_fixture()

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert result["baseline_rate"] == 0.1
    assert result["baseline_observations"] == 100
    assert result["alert_count"] == 0
    assert set(result["rows"]["observed_rate"]) == {0.1}
    assert set(result["rows"]["expected_conversions"]) == {10.0}
    assert all(result["rows"]["difference_ci_low"] <= 0)
    assert all(result["rows"]["difference_ci_high"] >= 0)
    assert not result["top_segment"]["is_anomaly"]
    assert "No segment passed" in result["summary"]


def test_genuine_anomaly_shows_correct_expected_observed_and_contribution():
    frame = make_ab_fixture(treatment_successes_by_segment={"Mobile": 30, "Web": 10})

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    mobile = result["rows"].set_index("segment").loc["Mobile"]

    assert result["top_segment"]["segment"] == "Mobile"
    assert result["baseline_rate"] == 0.1
    assert mobile["observations"] == 100
    assert mobile["observed_conversions"] == 30
    assert mobile["expected_conversions"] == 10.0
    assert mobile["observed_minus_expected_conversions"] == 20.0
    assert mobile["observed_rate"] == 0.3
    assert mobile["absolute_lift"] == pytest.approx(0.2)
    assert mobile["relative_lift_pct"] == pytest.approx(200.0)
    assert mobile["difference_ci_low"] > 0
    assert mobile["adjusted_p_value"] < 0.05
    assert mobile["is_anomaly"]
    assert result["alert_count"] == 1
    assert mobile["reference_label"] == "Control"
    assert mobile["observed_label"] == "Treatment"


def test_missing_outcomes_segments_and_exact_duplicates_are_reported_and_excluded():
    frame = make_ab_fixture(
        control_successes=6,
        control_n=60,
        treatment_successes_by_segment={"Mobile": 12, "Web": 12},
        treatment_n=30,
    )
    frame.loc[frame.index[-1], "converted"] = None
    frame.loc[frame.index[-2], "device"] = None
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    result = detect_business_anomaly(frame, segment_by="device", min_observations=10)

    mobile = result["rows"].set_index("segment").loc["Mobile"]
    assert result["baseline_observations"] == 61
    assert mobile["observations"] == 30
    assert mobile["observed_conversions"] == 12
    assert set(result["rows"]["segment"]) == {"Mobile", "Web"}
    assert result["rows"].set_index("segment").loc["Web", "observations"] == 28
    warning_text = "\n".join(result["warnings"])
    assert "Retained 1 exact duplicate row(s)" in warning_text
    assert "repeated user_id" in warning_text
    assert not result["inferential_valid"]
    assert result["alert_count"] == 0
    assert "missing or non-binary outcomes" in warning_text
    assert "missing 'device'" in warning_text


def test_identical_rows_without_user_id_are_retained_and_warned():
    rows = []
    for index in range(30):
        rows.append(
            {
                "group": "Control",
                "device": "Control",
                "converted": int(index < 3),
                "record": f"c-{index}",
            }
        )
    for device in ("Mobile", "Web"):
        for index in range(10):
            rows.append(
                {
                    "group": "Treatment",
                    "device": device,
                    "converted": int(index < 2),
                    "record": f"{device}-{index}",
                }
            )
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)

    result = detect_business_anomaly(frame, segment_by="device", min_observations=5)

    web = result["rows"].set_index("segment").loc["Web"]
    assert web["observations"] == 11
    assert any("Retained 1 identical row(s) without user_id" in warning for warning in result["warnings"])
    assert not result["inferential_valid"]
    assert result["alert_count"] == 0


def test_insufficient_reference_sample_returns_warning_not_anomaly():
    frame = make_ab_fixture(
        control_successes=2,
        control_n=10,
        treatment_successes_by_segment={"Mobile": 40},
        treatment_n=100,
    )

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert result["top_segment"] is None
    assert result["rows"].empty
    assert any("Reference population has 10" in warning for warning in result["warnings"])
    assert result["segment_accounting"]["excluded_segments"][0]["segment"] == "Mobile"


def test_low_volume_segment_is_excluded_from_testing_and_ranking():
    frame = make_ab_fixture(
        control_successes=10,
        control_n=100,
        treatment_successes_by_segment={"Mobile": 10, "Web": 10, "Tiny": 2},
        treatment_n=100,
    )
    tiny_rows = frame.index[frame["device"] == "Tiny"]
    frame = frame.drop(tiny_rows[29:])

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert "Tiny" not in result["rows"]["segment"].tolist()
    assert any("below the screening threshold" in warning for warning in result["warnings"])


def test_zero_reference_rate_uses_fisher_and_does_not_invent_relative_lift():
    frame = make_ab_fixture(
        control_successes=0,
        control_n=100,
        treatment_successes_by_segment={"Mobile": 2},
        treatment_n=100,
    )

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    mobile = result["rows"].iloc[0]

    assert mobile["expected_rate"] == 0
    assert mobile["relative_lift_pct"] is None
    assert mobile["test_method"] == "Fisher exact test"
    assert pd.isna(mobile["z_score"])
    assert 0 <= mobile["p_value"] <= 1


def test_fisher_exact_degenerate_rates_do_not_report_a_zero_z_score():
    frame = make_ab_fixture(
        control_successes=0,
        control_n=4,
        treatment_successes_by_segment={"Mobile": 4},
        treatment_n=4,
    )

    result = detect_business_anomaly(frame, segment_by="device", min_observations=1)
    mobile = result["rows"].iloc[0]

    assert mobile["test_method"] == "Fisher exact test"
    assert mobile["p_value"] < 0.05
    assert pd.isna(mobile["z_score"])


def test_bonferroni_family_size_does_not_shrink_with_direction_filter():
    frame = make_ab_fixture(
        treatment_successes_by_segment={
            "Mobile": 30,
            "Web": 10,
            "Tablet": 20,
        }
    )

    all_results = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    positive_results = detect_business_anomaly(
        frame,
        segment_by="device",
        min_observations=30,
        direction="positive",
    )

    all_p_values = all_results["rows"].set_index("segment")["adjusted_p_value"]
    assert len(all_results["rows"]) == 3
    assert len(positive_results["rows"]) == 2
    assert positive_results["rows"].set_index("segment").loc["Mobile", "adjusted_p_value"] == pytest.approx(
        all_p_values["Mobile"]
    )


def test_unassigned_dataset_uses_leave_one_segment_out_reference():
    frame = pd.DataFrame(
        {
            "device": ["A"] * 100 + ["B"] * 100,
            "converted": [1] * 20 + [0] * 80 + [1] * 10 + [0] * 90,
        }
    )

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    segment_a = result["rows"].set_index("segment").loc["A"]
    segment_b = result["rows"].set_index("segment").loc["B"]

    assert segment_a["expected_rate"] == 0.1
    assert segment_b["expected_rate"] == 0.2
    assert segment_a["reference_observations"] == 100
    assert any("leave-one-segment-out" in warning for warning in result["warnings"])


def test_repeated_users_are_flagged_as_non_independent_observations():
    frame = make_ab_fixture(
        control_successes=10,
        control_n=60,
        treatment_successes_by_segment={"Mobile": 20},
        treatment_n=60,
    )
    frame.loc[frame.index[-1], "user_id"] = frame.loc[frame.index[-60], "user_id"]

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert any("repeated user_id" in warning for warning in result["warnings"])
    assert any("does not establish causality" in warning for warning in result["warnings"])
    assert not result["inferential_valid"]
    assert result["alert_count"] == 0
    assert not result["rows"]["is_anomaly"].any()


def test_repeated_session_identifiers_suppress_inferential_alerts():
    frame = make_ab_fixture(
        control_successes=10,
        control_n=60,
        treatment_successes_by_segment={"Mobile": 30},
        treatment_n=60,
    )
    frame["session_id"] = [f"s-{index // 2}" for index in range(len(frame))]

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert any("repeated session_id" in warning for warning in result["warnings"])
    assert not result["inferential_valid"]
    assert result["alert_count"] == 0
    assert not result["rows"]["is_anomaly"].any()


def test_bonferroni_adjustment_uses_all_tested_segments():
    frame = make_ab_fixture(
        treatment_successes_by_segment={
            "Mobile": 30,
            "Web": 10,
            "Tablet": 10,
        }
    )

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)

    assert len(result["rows"]) == 3
    for _, row in result["rows"].iterrows():
        assert row["adjusted_p_value"] == pytest.approx(min(row["p_value"] * 3, 1.0))


def test_accounting_separates_included_segments_from_full_aggregate_change():
    frame = make_ab_fixture(
        control_successes=10,
        control_n=100,
        treatment_successes_by_segment={"Included": 20, "Small": 5},
        treatment_n=100,
    )
    small_rows = frame.index[frame["device"] == "Small"]
    frame = frame.drop(small_rows[29:])
    missing_category_rows = pd.DataFrame(
        {
            "user_id": [f"missing-{index}" for index in range(10)],
            "group": ["Treatment"] * 10,
            "device": [None] * 10,
            "converted": [1] * 5 + [0] * 5,
        }
    )
    frame = pd.concat([frame, missing_category_rows], ignore_index=True)

    result = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    accounting = result["segment_accounting"]

    assert accounting["included_segments"] == ["Included"]
    assert accounting["included_treatment_observations"] == 100
    assert accounting["total_treatment_observations"] == 139
    assert accounting["included_contribution"] == pytest.approx(10.0)
    assert accounting["full_aggregate_change"] == pytest.approx(16.1)
    assert accounting["unallocated_difference"] == pytest.approx(6.1)
    reasons = {entry["segment"]: entry["reason"] for entry in accounting["excluded_segments"]}
    assert "Insufficient valid observations" in reasons["Small"]
    assert "Missing 'device'" in reasons["(missing category)"]


def test_synthetic_demo_fixture_has_reproducible_detectable_signal():
    demo = generate_anomaly_demo_dataset(rows=1200, seed=7)
    result = detect_business_anomaly(demo, segment_by="device", min_observations=30)

    assert result["top_segment"] is not None
    assert result["top_segment"]["segment"] == "Mobile"
    assert result["top_segment"]["adjusted_p_value"] < 0.05
    assert result["top_segment"]["expected_conversions"] == (
        result["top_segment"]["expected_rate"] * result["top_segment"]["observations"]
    )


def test_minimum_observations_is_a_configurable_screening_cutoff():
    frame = make_ab_fixture(
        control_successes=3,
        control_n=10,
        treatment_successes_by_segment={"Mobile": 10},
        treatment_n=20,
    )

    screened = detect_business_anomaly(frame, segment_by="device", min_observations=30)
    lower_screen = detect_business_anomaly(frame, segment_by="device", min_observations=10)

    assert screened["top_segment"] is None
    assert lower_screen["top_segment"] is not None
    assert any("exploratory" in warning for warning in lower_screen["warnings"])
