from __future__ import annotations

import pandas as pd

from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint


def calculate_rate_confidence_interval(successes: int, total: int, alpha: float = 0.05) -> dict:
    if total <= 0:
        return {"low": float("nan"), "high": float("nan")}
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return {"low": float(low), "high": float(high)}


def calculate_difference_confidence_interval(control_df, treatment_df, alpha: float = 0.05) -> dict:
    control_values = pd.to_numeric(control_df["converted"], errors="coerce").dropna()
    treatment_values = pd.to_numeric(treatment_df["converted"], errors="coerce").dropna()
    control_values = control_values[control_values.isin([0, 1])]
    treatment_values = treatment_values[treatment_values.isin([0, 1])]
    control_successes = int(control_values.sum())
    treatment_successes = int(treatment_values.sum())
    control_total = len(control_values)
    treatment_total = len(treatment_values)
    if not control_total or not treatment_total:
        return {"low": float("nan"), "high": float("nan")}
    low, high = confint_proportions_2indep(
        treatment_successes,
        treatment_total,
        control_successes,
        control_total,
        alpha=alpha,
        method="newcomb",
    )
    return {"low": float(low), "high": float(high), "method": "Newcombe-Wilson"}
