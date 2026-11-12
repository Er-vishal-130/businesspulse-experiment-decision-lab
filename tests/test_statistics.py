import pandas as pd

from modules.confidence_intervals import calculate_difference_confidence_interval
from modules.effect_size import calculate_effect_size
from modules.hypothesis_testing import run_two_proportion_test
from modules.report_generator import build_html_report, build_summary_text


def test_two_proportion_test_returns_values():
    control = pd.DataFrame({"converted": [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]})
    treatment = pd.DataFrame({"converted": [0, 1, 1, 1, 1, 1, 1, 0, 1, 1]})
    result = run_two_proportion_test(control, treatment)
    assert "p_value" in result
    assert "decision" in result
    assert result["p_value"] >= 0.0
    assert result["method"] == "Fisher's exact test"


def test_large_sample_test_reports_treatment_minus_control_interval():
    control = pd.DataFrame({"converted": [0] * 900 + [1] * 100})
    treatment = pd.DataFrame({"converted": [0] * 850 + [1] * 150})

    result = run_two_proportion_test(control, treatment)
    interval = calculate_difference_confidence_interval(control, treatment)

    assert result["method"] == "Two-proportion z-test"
    assert interval["low"] > 0
    assert interval["high"] > interval["low"]


def test_missing_arm_is_not_reported_as_a_test_result():
    result = run_two_proportion_test(
        pd.DataFrame({"converted": [0, 1]}),
        pd.DataFrame({"converted": []}),
    )
    assert result["decision"] == "Not estimable"
    assert pd.isna(result["p_value"])
    assert result["warnings"]


def test_repeated_identifiers_suppress_independent_row_test():
    control = pd.DataFrame(
        {
            "user_id": ["u-1", "u-2", "u-1", "u-4"],
            "converted": [0, 0, 1, 0],
        }
    )
    treatment = pd.DataFrame(
        {"user_id": ["t-1", "t-2", "t-3", "t-4"], "converted": [1, 1, 0, 1]}
    )

    result = run_two_proportion_test(control, treatment)

    assert result["decision"] == "Not estimable"
    assert result["method"] == "Not estimable"
    assert pd.isna(result["p_value"])
    assert any("repeated user_id" in warning for warning in result["warnings"])


def test_unique_or_missing_identifiers_do_not_prove_independence():
    control = pd.DataFrame({"user_id": ["c-1", "c-2"], "converted": [0, 1]})
    treatment = pd.DataFrame({"user_id": ["t-1", "t-2"], "converted": [1, 1]})

    result = run_two_proportion_test(control, treatment)

    assert any("unique identifiers do not prove independence" in warning for warning in result["warnings"])

    no_ids = run_two_proportion_test(
        pd.DataFrame({"converted": [0, 1]}),
        pd.DataFrame({"converted": [1, 1]}),
    )
    assert any("unique identifiers do not prove independence" in warning for warning in no_ids["warnings"])


def test_reports_label_unsupported_inference_as_not_estimable():
    summary = {
        "control_conversion": 0.1,
        "treatment_conversion": 0.2,
        "absolute_lift": 0.1,
        "relative_lift": 100.0,
    }
    hypothesis = {"p_value": float("nan")}

    text = build_summary_text(
        summary,
        float("nan"),
        {"low": float("nan"), "high": float("nan")},
        "INCONCLUSIVE",
        "No supported recommendation.",
    )
    html = build_html_report(summary, hypothesis, "No supported recommendation.", "INCONCLUSIVE")

    assert "p-value: Not estimable" in text
    assert "95% CI for difference: Not estimable" in text
    assert "<li>p-value: Not estimable</li>" in html


def test_confidence_interval_and_effect_size():
    control = pd.DataFrame({"converted": [0, 1, 0, 0, 1, 0]})
    treatment = pd.DataFrame({"converted": [1, 1, 1, 1, 1, 1]})
    ci = calculate_difference_confidence_interval(control, treatment)
    effect = calculate_effect_size(control, treatment)
    assert "low" in ci and "high" in ci
    assert "difference_in_proportions" in effect
