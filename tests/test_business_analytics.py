import pandas as pd

from modules.ask_data import answer_question
from modules.business_analytics import summarize_time_trend
from modules.metrics import summarize_group_metrics
from modules.preprocessing import filter_dataset
from modules.validator import clean_dataframe, validate_dataframe


def test_minimal_experiment_schema_and_optional_kpis():
    frame = pd.DataFrame(
        {
            "group": ["Control", "Treatment"],
            "converted": [0, 1],
            "session_id": ["s1", "s2"],
        }
    )

    assert validate_dataframe(frame)["status"] == "Valid"
    summary = summarize_group_metrics(frame)
    assert summary["total_users"] == 2
    assert summary["total_sessions"] == 2
    assert summary["total_conversions"] == 1
    assert summary["revenue"] is None


def test_cleaning_does_not_turn_invalid_revenue_into_zero():
    frame = pd.DataFrame(
        {
            "group": ["Control", "Treatment"],
            "converted": ["0", "invalid"],
            "revenue": ["not available", "15"],
        }
    )

    cleaned = clean_dataframe(frame)
    assert len(cleaned) == 1
    assert pd.isna(cleaned.loc[0, "revenue"])
    assert cleaned.loc[0, "converted"] == 0


def test_time_trend_uses_observed_daily_group_rates():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01 10:00", "2025-01-01 12:00", "2025-01-02"],
            "group": ["Control", "Control", "Treatment"],
            "converted": [0, 1, 1],
        }
    )

    trend = summarize_time_trend(frame)
    assert len(trend) == 2
    assert trend.loc[trend["group"] == "Control", "conversion_rate"].iloc[0] == 0.5
    assert trend.loc[trend["group"] == "Treatment", "conversions"].iloc[0] == 1


def test_ask_data_answers_from_observed_segments_and_reports_missing_columns():
    frame = pd.DataFrame(
        {
            "group": ["Control", "Treatment", "Control", "Treatment"],
            "converted": [1, 1, 0, 0],
            "device": ["Mobile", "Mobile", "Web", "Web"],
            "revenue": [10, 20, 0, 0],
        }
    )

    answer = answer_question(frame, "Which device has the highest conversion rate?")
    assert "Mobile" in answer["answer"]
    assert "2/2" in answer["answer"]
    assert answer["table"].iloc[0]["conversion_rate"] == 1.0

    missing = answer_question(frame.drop(columns="device"), "Which device has the highest conversion?")
    assert "no device" in missing["answer"]


def test_date_filter_includes_entire_end_date_and_empty_selections():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 23:59", "2025-01-02 23:59"]),
            "group": ["Control", "Treatment"],
        }
    )

    filtered = filter_dataset(frame, date_max="2025-01-01")
    assert len(filtered) == 1
    assert filter_dataset(frame, group=[]).empty


def test_selecting_all_categories_preserves_missing_categories_for_accounting():
    frame = pd.DataFrame(
        {
            "group": ["Control", "Treatment", "Treatment"],
            "device": ["Mobile", "Mobile", None],
            "converted": [0, 1, 1],
        }
    )

    all_selected = filter_dataset(
        frame,
        group=["Control", "Treatment"],
        device=["Mobile"],
    )
    selected_subset = filter_dataset(frame, device=[])

    assert len(all_selected) == 3
    assert all_selected["device"].isna().sum() == 1
    assert selected_subset.empty
