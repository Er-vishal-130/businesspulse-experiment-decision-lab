from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from statsmodels.stats.proportion import confint_proportions_2indep


def generate_anomaly_demo_dataset(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Create a synthetic dataset with a deliberate conversion anomaly for demos/tests."""
    rng = np.random.default_rng(seed)
    device_choices = ["Mobile", "Web", "Tablet"]
    source_choices = ["Organic", "Paid Search", "Social", "Referral", "Email"]
    page_choices = ["Landing", "Pricing", "Checkout", "Support"]
    country_choices = ["India", "USA", "UK", "Germany"]
    dates = pd.date_range("2025-01-01", periods=rows, freq="h")

    groups = rng.choice(["Control", "Treatment"], size=rows)
    devices = rng.choice(device_choices, size=rows, p=[0.5, 0.4, 0.1])
    sources = rng.choice(source_choices, size=rows)
    countries = rng.choice(country_choices, size=rows)
    pages = rng.choice(page_choices, size=rows)
    conversion_probabilities = np.where(
        (groups == "Treatment") & (devices == "Mobile"),
        0.30,
        0.09,
    )
    conversions = rng.binomial(1, conversion_probabilities)

    data: list[dict[str, Any]] = []
    for idx in range(rows):
        converted = int(conversions[idx])
        data.append(
            {
                "user_id": idx + 1,
                "group": groups[idx],
                "converted": converted,
                "revenue": 120.0 if converted else 0.0,
                "device": devices[idx],
                "traffic_source": sources[idx],
                "country": countries[idx],
                "page_type": pages[idx],
                "date": dates[idx],
                "session_duration": 30 + (idx % 120),
            }
        )
    return pd.DataFrame(data)


def _valid_outcomes(df: pd.DataFrame, metric_col: str) -> pd.Series:
    values = pd.to_numeric(df[metric_col], errors="coerce")
    return values[values.isin([0, 1])]


def _empty_result(
    segment_by: str | None,
    warning: str,
    baseline_rate: float = 0.0,
    baseline_observations: int = 0,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "segment_column": segment_by,
        "baseline_rate": baseline_rate,
        "baseline_observations": baseline_observations,
        "baseline_label": None,
        "observed_label": None,
        "warnings": [*(warnings or []), warning],
        "top_segment": None,
        "rows": pd.DataFrame(),
        "alert_count": 0,
        "inferential_valid": False,
        "segment_accounting": {
            "included_segments": [],
            "excluded_segments": [],
            "included_treatment_observations": 0,
            "total_treatment_observations": 0,
            "included_contribution": None,
            "full_aggregate_change": None,
            "unallocated_difference": None,
        },
        "summary": warning,
    }


def _compare_binary_rates(
    reference_conversions: int,
    reference_observations: int,
    observed_conversions: int,
    observed_observations: int,
) -> tuple[float, float, str]:
    """Return a two-sided independent two-proportion test statistic, p-value, and method."""
    reference_rate = reference_conversions / reference_observations
    observed_rate = observed_conversions / observed_observations
    lift = observed_rate - reference_rate

    pooled_rate = (
        reference_conversions + observed_conversions
    ) / (reference_observations + observed_observations)
    expected = [
        reference_observations * pooled_rate,
        reference_observations * (1 - pooled_rate),
        observed_observations * pooled_rate,
        observed_observations * (1 - pooled_rate),
    ]
    if min(expected) < 5:
        _, p_value = fisher_exact(
            [
                [reference_conversions, reference_observations - reference_conversions],
                [observed_conversions, observed_observations - observed_conversions],
            ],
            alternative="two-sided",
        )
        return float("nan"), float(p_value), "Fisher exact test"

    pooled_se = math.sqrt(
        pooled_rate * (1 - pooled_rate)
        * (1 / reference_observations + 1 / observed_observations)
    )
    z_score = lift / pooled_se if pooled_se else 0.0
    p_value = float(2 * norm.sf(abs(z_score)))
    return z_score, p_value, "Two-proportion score test"


def detect_business_anomaly(
    df: pd.DataFrame,
    segment_by: str | None = None,
    metric_col: str = "converted",
    min_observations: int = 30,
    direction: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare disjoint binary-outcome populations and rank segment deviations.

    With Control and Treatment groups, Control is the reference population and
    Treatment segments are compared against it. Otherwise, each segment is compared
    with all other observed rows (leave-one-segment-out reference). P-values are
    exploratory and Bonferroni-adjusted across tested segments. Repeated user_id or
    session_id values prevent inferential alerts because independence is unverified.
    """
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if direction not in (None, "positive", "negative"):
        raise ValueError("direction must be None, 'positive', or 'negative'.")
    if df is None or df.empty:
        return _empty_result(segment_by, "No data available for anomaly detection.")
    if metric_col not in df.columns:
        return _empty_result(segment_by, f"Metric column '{metric_col}' is missing.")

    frame = df.copy()
    duplicate_rows = frame.duplicated()
    warnings: list[str] = []
    if duplicate_rows.any():
        if "user_id" in frame.columns:
            identified_duplicates = duplicate_rows & frame["user_id"].notna()
            unidentified_duplicates = duplicate_rows & frame["user_id"].isna()
            if identified_duplicates.any():
                warnings.append(
                    f"Retained {int(identified_duplicates.sum())} exact duplicate row(s) with a known user_id; "
                    "repeated observations are not automatically removed."
                )
            if unidentified_duplicates.any():
                warnings.append(
                    f"Retained {int(unidentified_duplicates.sum())} identical row(s) with missing user_id because duplicate identity cannot be established."
                )
        else:
            warnings.append(
                f"Retained {int(duplicate_rows.sum())} identical row(s) without user_id because duplicate identity cannot be established."
            )

    numeric_outcomes = pd.to_numeric(frame[metric_col], errors="coerce")
    invalid_outcomes = int((~numeric_outcomes.isin([0, 1])).sum())
    frame[metric_col] = numeric_outcomes
    frame = frame[numeric_outcomes.isin([0, 1])].copy()
    if invalid_outcomes:
        warnings.append(
            f"Excluded {invalid_outcomes} row(s) with missing or non-binary outcomes."
        )
    if frame.empty:
        return _empty_result(
            segment_by,
            "No valid binary outcomes available for anomaly detection.",
            warnings=warnings,
        )

    candidate_columns = [
        "device",
        "traffic_source",
        "country",
        "gender",
        "page_type",
        "experiment_id",
    ]
    if segment_by is None:
        segment_by = next(
            (column for column in candidate_columns if column in frame.columns),
            None,
        )
    if segment_by is None or segment_by not in frame.columns:
        return _empty_result(
            segment_by,
            "No suitable segment column is available for anomaly detection.",
            warnings=warnings,
        )

    use_experiment_reference = False
    if "group" in frame.columns:
        normalized_groups = frame["group"].astype(str).str.strip().str.casefold()
        control_mask = normalized_groups == "control"
        treatment_mask = normalized_groups == "treatment"
        use_experiment_reference = bool(control_mask.any() and treatment_mask.any())

    if use_experiment_reference:
        reference = frame[control_mask].copy()
        observed_population = frame[treatment_mask].copy()
        baseline_label = "Control"
        observed_label = "Treatment"
        ignored_group_rows = int((~(control_mask | treatment_mask)).sum())
        if ignored_group_rows:
            warnings.append(
                f"Excluded {ignored_group_rows} row(s) outside the Control/Treatment groups."
            )
    else:
        reference = frame
        observed_population = frame
        baseline_label = "Other rows in dataset"
        observed_label = "Segment"
        warnings.append(
            "No complete Control/Treatment comparison is available; using a leave-one-segment-out reference."
        )

    dependency_counts: dict[str, int] = {}
    for identifier in ("user_id", "session_id"):
        if identifier in frame.columns:
            repeated_count = int(frame[identifier].dropna().duplicated().sum())
            if repeated_count:
                dependency_counts[identifier] = repeated_count
                warnings.append(
                    f"Found {repeated_count} repeated {identifier} row(s); inferential alerts are suppressed without a design-specific method."
                )
    possible_duplicate_observations = bool(duplicate_rows.any())
    if possible_duplicate_observations:
        warnings.append(
            "Inferential alerts are suppressed because exact duplicate rows may represent dependent observations."
        )
    inferential_valid = not dependency_counts and not possible_duplicate_observations

    baseline_outcomes = _valid_outcomes(reference, metric_col)
    baseline_n = len(baseline_outcomes)
    if baseline_n < min_observations:
        excluded = []
        observed_all = _valid_outcomes(observed_population, metric_col)
        if segment_by in observed_population.columns:
            for value, segment_df in observed_population.groupby(
                segment_by, dropna=True, observed=True
            ):
                segment_n = len(_valid_outcomes(segment_df, metric_col))
                excluded.append(
                    {
                        "segment": str(value),
                        "reason": f"Reference population has {baseline_n} valid observations; screening minimum is {min_observations}.",
                        "observations": segment_n,
                    }
                )
            missing_n = len(
                _valid_outcomes(
                    observed_population.loc[observed_population[segment_by].isna()],
                    metric_col,
                )
            )
            if missing_n:
                excluded.append(
                    {
                        "segment": "(missing category)",
                        "reason": f"Missing '{segment_by}' value.",
                        "observations": missing_n,
                    }
                )
        result = _empty_result(
            segment_by,
            f"Reference population has {baseline_n} valid observations; at least {min_observations} are required.",
            baseline_observations=baseline_n,
            warnings=warnings,
        )
        result["segment_accounting"]["excluded_segments"] = excluded
        result["segment_accounting"]["total_treatment_observations"] = len(observed_all)
        return result
    baseline_rate = float(baseline_outcomes.mean())
    observed_outcomes_all = _valid_outcomes(observed_population, metric_col)
    missing_segment_mask = observed_population[segment_by].isna()
    missing_segment_n = len(
        _valid_outcomes(observed_population.loc[missing_segment_mask], metric_col)
    )
    if missing_segment_n:
        warnings.append(
            f"Excluded {missing_segment_n} observed-population row(s) with missing '{segment_by}' categories."
        )

    rows: list[dict[str, Any]] = []
    excluded_segments: list[dict[str, Any]] = []
    included_segments: list[str] = []
    included_observations = 0
    included_contribution = 0.0
    for value, raw_segment_df in observed_population.groupby(segment_by, dropna=True, observed=True):
        if use_experiment_reference:
            segment_df = raw_segment_df[
                raw_segment_df["group"].astype(str).str.strip().str.casefold() == "treatment"
            ]
            segment_reference_outcomes = baseline_outcomes
        else:
            segment_df = raw_segment_df
            segment_reference_outcomes = _valid_outcomes(
                frame[frame[segment_by] != value], metric_col
            )

        observed_outcomes = _valid_outcomes(segment_df, metric_col)
        observed_n = len(observed_outcomes)
        reference_n = len(segment_reference_outcomes)
        if observed_n < min_observations or reference_n < min_observations:
            excluded_segments.append(
                {
                    "segment": str(value),
                    "reason": f"Insufficient valid observations (observed n={observed_n}, reference n={reference_n}, screening minimum={min_observations}).",
                    "observations": observed_n,
                }
            )
            continue

        segment_reference_rate = float(segment_reference_outcomes.mean())
        observed_conversions = int(observed_outcomes.sum())
        reference_conversions = int(segment_reference_outcomes.sum())
        observed_rate = observed_conversions / observed_n
        absolute_change = observed_rate - segment_reference_rate
        z_score, p_value, test_method = _compare_binary_rates(
            reference_conversions,
            reference_n,
            observed_conversions,
            observed_n,
        )
        ci_low, ci_high = confint_proportions_2indep(
            observed_conversions,
            observed_n,
            reference_conversions,
            reference_n,
            alpha=0.05,
            method="newcomb",
        )
        contribution = observed_conversions - segment_reference_rate * observed_n
        included_segments.append(str(value))
        included_observations += observed_n
        included_contribution += contribution
        rows.append(
            {
                "segment": str(value),
                "segment_column": segment_by,
                "reference_label": baseline_label,
                "observed_label": observed_label,
                "reference_observations": reference_n,
                "observations": observed_n,
                "observed_conversions": observed_conversions,
                "expected_conversions": segment_reference_rate * observed_n,
                "observed_minus_expected_conversions": contribution,
                "observed_rate": observed_rate,
                "expected_rate": segment_reference_rate,
                "absolute_lift": absolute_change,
                "relative_lift_pct": (
                    absolute_change / segment_reference_rate * 100.0
                    if segment_reference_rate > 0
                    else None
                ),
                "z_score": z_score,
                "p_value": p_value,
                "test_method": test_method,
                "difference_ci_low": float(ci_low),
                "difference_ci_high": float(ci_high),
            }
        )

    if excluded_segments:
        warnings.append(f"Excluded {len(excluded_segments)} segment(s) below the screening threshold.")
    if missing_segment_n:
        excluded_segments.append(
            {
                "segment": "(missing category)",
                "reason": f"Missing '{segment_by}' value.",
                "observations": missing_segment_n,
            }
        )
    if use_experiment_reference:
        total_treatment_n = len(observed_outcomes_all)
        total_treatment_conversions = int(observed_outcomes_all.sum())
        full_aggregate_change = (
            total_treatment_conversions - baseline_rate * total_treatment_n
        )
        unallocated_difference = full_aggregate_change - included_contribution
    else:
        total_treatment_n = len(observed_outcomes_all)
        full_aggregate_change = None
        unallocated_difference = None
    segment_accounting = {
        "included_segments": included_segments,
        "excluded_segments": excluded_segments,
        "included_treatment_observations": included_observations,
        "total_treatment_observations": total_treatment_n,
        "included_contribution": included_contribution if use_experiment_reference else None,
        "full_aggregate_change": full_aggregate_change,
        "unallocated_difference": unallocated_difference,
    }
    if not rows:
        result = _empty_result(
            segment_by,
            f"No segment has at least {min_observations} valid observations.",
            baseline_rate=baseline_rate,
            baseline_observations=baseline_n,
            warnings=warnings,
        )
        result["segment_accounting"] = segment_accounting
        return result

    segment_table = pd.DataFrame(rows)
    comparison_count = len(segment_table)
    segment_table["adjusted_p_value"] = (
        segment_table["p_value"] * comparison_count
    ).clip(upper=1.0)
    if direction == "positive":
        segment_table = segment_table[segment_table["absolute_lift"] > 0].copy()
    elif direction == "negative":
        segment_table = segment_table[segment_table["absolute_lift"] < 0].copy()
    if segment_table.empty:
        result = _empty_result(
            segment_by,
            "No segments meet the requested direction filter.",
            baseline_rate=baseline_rate,
            baseline_observations=baseline_n,
            warnings=warnings,
        )
        result["segment_accounting"] = segment_accounting
        return result

    segment_table["is_anomaly"] = (
        (segment_table["adjusted_p_value"] < alpha) if inferential_valid else False
    )
    segment_table = segment_table.sort_values(
        ["adjusted_p_value", "absolute_lift", "observations"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    top_segment = segment_table.iloc[0].to_dict()
    top_segment["direction"] = "above" if top_segment["absolute_lift"] >= 0 else "below"
    alert_count = int(segment_table["is_anomaly"].sum())
    if not inferential_valid:
        summary = (
            "Inferential alerts suppressed because repeated identifiers indicate potentially dependent observations. "
            "Nominal adjusted p-values are diagnostic only."
        )
    elif alert_count:
        summary = (
            f"{alert_count} segment(s) show a statistically supported difference after "
            f"multiple-comparison adjustment. Strongest evidence: {top_segment['segment']} "
            f"is {abs(top_segment['absolute_lift']):.2%} {top_segment['direction']} "
            f"{baseline_label} ({baseline_rate:.2%})."
        )
    else:
        summary = (
            f"No segment passed the adjusted significance threshold (alpha={alpha:.2f}). "
            f"Largest observed deviation: {top_segment['segment']} is "
            f"{abs(top_segment['absolute_lift']):.2%} {top_segment['direction']} "
            f"{baseline_label} ({baseline_rate:.2%}); treat this as exploratory, not an alert."
        )

    warnings.append(
        "Segment comparisons are exploratory; adjusted significance does not establish causality or adequate statistical power."
    )

    return {
        "segment_column": segment_by,
        "baseline_rate": baseline_rate,
        "baseline_observations": baseline_n,
        "baseline_label": baseline_label,
        "observed_label": observed_label,
        "warnings": warnings,
        "top_segment": top_segment,
        "rows": segment_table,
        "alert_count": alert_count,
        "inferential_valid": inferential_valid,
        "segment_accounting": segment_accounting,
        "summary": summary,
    }
