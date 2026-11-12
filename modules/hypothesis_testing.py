from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest


def run_two_proportion_test(control_df, treatment_df, alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if alternative == "one-sided":
        alternative = "greater"
    if alternative not in {"two-sided", "greater", "smaller"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'smaller'.")

    control_values = (
        pd.to_numeric(control_df["converted"], errors="coerce").dropna()
        if "converted" in control_df
        else pd.Series(dtype=float)
    )
    treatment_values = (
        pd.to_numeric(treatment_df["converted"], errors="coerce").dropna()
        if "converted" in treatment_df
        else pd.Series(dtype=float)
    )
    if not control_values.isin([0, 1]).all() or not treatment_values.isin([0, 1]).all():
        raise ValueError("Conversion outcomes must contain only 0/1 values.")
    control_total = int(len(control_values))
    treatment_total = int(len(treatment_values))
    control_successes = int(control_values.sum())
    treatment_successes = int(treatment_values.sum())
    warnings = [
        "The test assumes independent randomized units; unique identifiers do not prove independence."
    ]
    if min(control_total, treatment_total) < 30:
        warnings.append("Fewer than 30 valid observations in at least one group; interpret estimates cautiously.")

    if not control_total or not treatment_total:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "alpha": float(alpha),
            "decision": "Not estimable",
            "interpretation": "Both Control and Treatment need valid binary conversion observations.",
            "control_successes": control_successes,
            "treatment_successes": treatment_successes,
            "control_total": control_total,
            "treatment_total": treatment_total,
            "method": "Not estimable",
            "warnings": warnings,
        }

    combined = pd.concat([control_df, treatment_df], ignore_index=True, sort=False)
    has_repeated_identifiers = False
    for identifier in ("user_id", "session_id"):
        if identifier in combined.columns:
            repeated_count = int(
                combined[identifier].dropna().astype(str).duplicated().sum()
            )
            if repeated_count:
                has_repeated_identifiers = True
                warnings.append(
                    f"Found {repeated_count} repeated {identifier} row(s); "
                    "the independent-row test is unavailable without a design-specific method."
                )
    if has_repeated_identifiers:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "alpha": float(alpha),
            "decision": "Not estimable",
            "interpretation": (
                "Inferential testing was suppressed because repeated identifiers may indicate "
                "dependent observations; use a method appropriate to the study design."
            ),
            "control_successes": control_successes,
            "treatment_successes": treatment_successes,
            "control_total": control_total,
            "treatment_total": treatment_total,
            "method": "Not estimable",
            "warnings": warnings,
        }

    if max(control_total, treatment_total) / min(control_total, treatment_total) > 2:
        warnings.append("Group sizes differ by more than 2:1; check randomization and assignment.")

    pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
    expected = np.array(
        [
            [treatment_total * pooled_rate, treatment_total * (1 - pooled_rate)],
            [control_total * pooled_rate, control_total * (1 - pooled_rate)],
        ]
    )
    if (expected < 5).any():
        method = "Fisher's exact test"
        table = [
            [treatment_successes, treatment_total - treatment_successes],
            [control_successes, control_total - control_successes],
        ]
        fisher_alternative = "greater" if alternative == "greater" else "less" if alternative == "smaller" else "two-sided"
        result = fisher_exact(table, alternative=fisher_alternative)
        stat, p_value = float(result.statistic), float(result.pvalue)
        warnings.append("Expected cell count below 5; using Fisher's exact test instead of the large-sample z-test.")
    else:
        method = "Two-proportion z-test"
        z_alternative = "larger" if alternative == "greater" else "smaller" if alternative == "smaller" else "two-sided"
        stat, p_value = proportions_ztest(
            np.array([treatment_successes, control_successes]),
            np.array([treatment_total, control_total]),
            alternative=z_alternative,
        )
        stat, p_value = float(stat), float(p_value)

    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"
    interpretation = (
        "Under the independent-randomized-unit assumption, the data provide evidence of a difference at the selected alpha level. This does not by itself establish business value or causality."
        if p_value < alpha
        else "Under the independent-randomized-unit assumption, the data do not provide enough evidence to distinguish the groups at the selected alpha level; this is not proof that the groups are equivalent."
    )

    return {
        "statistic": stat,
        "p_value": p_value,
        "alpha": float(alpha),
        "decision": decision,
        "interpretation": interpretation,
        "control_successes": control_successes,
        "treatment_successes": treatment_successes,
        "control_total": control_total,
        "treatment_total": treatment_total,
        "method": method,
        "warnings": warnings,
    }
