import pandas as pd

from modules.metrics import (
    build_business_scenario_table,
    calculate_absolute_lift,
    calculate_conversion_rate,
    calculate_experiment_health_score,
    calculate_relative_lift,
)


def test_conversion_rate():
    series = pd.Series([0, 1, 1, 0, 1])
    assert abs(calculate_conversion_rate(series) - 0.6) < 1e-9


def test_absolute_lift_and_relative_lift():
    assert abs(calculate_absolute_lift(0.1, 0.15) - 0.05) < 1e-9
    assert abs(calculate_relative_lift(0.1, 0.15) - 50.0) < 1e-9


def test_health_score_and_scenario_table():
    summary = {"total_users": 1000, "absolute_lift": 0.04}
    score = calculate_experiment_health_score(summary, 0.02, alpha=0.05)
    scenario_df = build_business_scenario_table(50000, 0.02, 120, [0.5, 1.0, 2.0])
    assert 0 <= score <= 100
    assert len(scenario_df) == 3
