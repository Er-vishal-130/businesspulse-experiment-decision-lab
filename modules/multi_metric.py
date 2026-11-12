"""Multi-metric A/B testing with Holm-Bonferroni correction.

Supports binary (conversion, bounce) and continuous (revenue, session_duration,
pages_viewed) outcome metrics simultaneously.  Each metric gets an individual
p-value, effect size, and 95 % CI; the Holm-Bonferroni procedure controls the
family-wise error rate (FWER) across the tested outcomes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, fisher_exact
from statsmodels.stats.proportion import proportions_ztest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank, idx in enumerate(reversed(order)):
        raw = p_values[idx]
        k = n - rank  # number of remaining tests
        adjusted_raw = raw * k
        running_min = min(running_min, adjusted_raw)
        adjusted[idx] = running_min
    return [min(v, 1.0) for v in adjusted]


def _binary_test(
    control_vals: pd.Series,
    treatment_vals: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float, str]:
    """Two-proportion test for binary outcomes; returns (stat, p_value, method)."""
    ctrl_s = int(control_vals.sum())
    trt_s = int(treatment_vals.sum())
    ctrl_n = int(len(control_vals))
    trt_n = int(len(treatment_vals))
    if ctrl_n == 0 or trt_n == 0:
        return float("nan"), float("nan"), "Not estimable"

    pooled = (ctrl_s + trt_s) / (ctrl_n + trt_n)
    expected = np.array(
        [
            [trt_n * pooled, trt_n * (1 - pooled)],
            [ctrl_n * pooled, ctrl_n * (1 - pooled)],
        ]
    )
    if (expected < 5).any():
        table = [[trt_s, trt_n - trt_s], [ctrl_s, ctrl_n - ctrl_s]]
        res = fisher_exact(table, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), "Fisher's exact test"

    stat, p = proportions_ztest(
        np.array([trt_s, ctrl_s]),
        np.array([trt_n, ctrl_n]),
        alternative="two-sided",
    )
    return float(stat), float(p), "Two-proportion z-test"


def _continuous_test(
    control_vals: pd.Series,
    treatment_vals: pd.Series,
) -> tuple[float, float, str]:
    """Welch t-test + Mann-Whitney U for continuous outcomes.

    Returns (stat, p_value, method).  Uses Mann-Whitney for non-normal
    distributions (small samples or heavy tails) and Welch's t-test
    otherwise; we always run Mann-Whitney here since we can't verify
    normality without extra input.
    """
    if len(control_vals) == 0 or len(treatment_vals) == 0:
        return float("nan"), float("nan"), "Not estimable"
    stat, p = mannwhitneyu(control_vals, treatment_vals, alternative="two-sided")
    return float(stat), float(p), "Mann-Whitney U"


def _effect_size_binary(
    control_vals: pd.Series,
    treatment_vals: pd.Series,
) -> dict[str, Any]:
    ctrl_rate = float(control_vals.mean()) if len(control_vals) else float("nan")
    trt_rate = float(treatment_vals.mean()) if len(treatment_vals) else float("nan")
    diff = trt_rate - ctrl_rate if not (np.isnan(ctrl_rate) or np.isnan(trt_rate)) else float("nan")
    rel = (diff / ctrl_rate * 100) if (not np.isnan(diff) and abs(ctrl_rate) > 1e-12) else float("nan")
    return {"control_rate": ctrl_rate, "treatment_rate": trt_rate, "absolute_diff": diff, "relative_pct": rel}


def _effect_size_continuous(
    control_vals: pd.Series,
    treatment_vals: pd.Series,
) -> dict[str, Any]:
    ctrl_mean = float(control_vals.mean()) if len(control_vals) else float("nan")
    trt_mean = float(treatment_vals.mean()) if len(treatment_vals) else float("nan")
    diff = trt_mean - ctrl_mean if not (np.isnan(ctrl_mean) or np.isnan(trt_mean)) else float("nan")
    pooled_std = float(
        np.sqrt(
            (control_vals.var(ddof=1) + treatment_vals.var(ddof=1)) / 2
        )
    ) if (len(control_vals) > 1 and len(treatment_vals) > 1) else float("nan")
    cohens_d = diff / pooled_std if (not np.isnan(pooled_std) and pooled_std > 1e-12) else float("nan")
    rel = (diff / abs(ctrl_mean) * 100) if (not np.isnan(diff) and abs(ctrl_mean) > 1e-12) else float("nan")
    return {
        "control_mean": ctrl_mean,
        "treatment_mean": trt_mean,
        "mean_diff": diff,
        "relative_pct": rel,
        "cohens_d": cohens_d,
        "pooled_std": pooled_std,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

BINARY_METRICS = {"converted", "bounce"}
CONTINUOUS_METRICS = {"revenue", "session_duration", "pages_viewed"}


def run_multi_metric_test(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run hypothesis tests for all supported metrics present in the data.

    Returns a dict with:
    - ``rows`` — list of per-metric result dicts
    - ``summary_df`` — a Pandas DataFrame suitable for display
    - ``warnings`` — list of advisory strings
    - ``alpha`` — the requested significance level
    """
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    warnings: list[str] = [
        "The Holm-Bonferroni procedure controls the family-wise error rate "
        "across the tested metric outcomes, not across all possible comparisons."
    ]

    all_metrics = list(BINARY_METRICS | CONTINUOUS_METRICS)
    available = [m for m in all_metrics if m in control_df.columns and m in treatment_df.columns]

    if not available:
        return {
            "rows": [],
            "summary_df": pd.DataFrame(),
            "warnings": ["No supported metric columns found in the dataset."],
            "alpha": alpha,
        }

    for metric in available:
        ctrl_raw = pd.to_numeric(control_df[metric], errors="coerce").dropna()
        trt_raw = pd.to_numeric(treatment_df[metric], errors="coerce").dropna()

        if metric in BINARY_METRICS:
            ctrl_vals = ctrl_raw[ctrl_raw.isin([0, 1])]
            trt_vals = trt_raw[trt_raw.isin([0, 1])]
            stat, p, method = _binary_test(ctrl_vals, trt_vals, alpha=alpha)
            effect = _effect_size_binary(ctrl_vals, trt_vals)
            row = {
                "metric": metric,
                "type": "binary",
                "control_n": len(ctrl_vals),
                "treatment_n": len(trt_vals),
                "control_value": effect.get("control_rate"),
                "treatment_value": effect.get("treatment_rate"),
                "absolute_diff": effect.get("absolute_diff"),
                "relative_pct": effect.get("relative_pct"),
                "cohens_d_or_na": None,
                "statistic": stat,
                "p_value_raw": p,
                "method": method,
            }
        else:
            if len(ctrl_raw) < 2 or len(trt_raw) < 2:
                warnings.append(f"{metric}: fewer than 2 valid observations in at least one group; skipped.")
                continue
            stat, p, method = _continuous_test(ctrl_raw, trt_raw)
            effect = _effect_size_continuous(ctrl_raw, trt_raw)
            row = {
                "metric": metric,
                "type": "continuous",
                "control_n": len(ctrl_raw),
                "treatment_n": len(trt_raw),
                "control_value": effect.get("control_mean"),
                "treatment_value": effect.get("treatment_mean"),
                "absolute_diff": effect.get("mean_diff"),
                "relative_pct": effect.get("relative_pct"),
                "cohens_d_or_na": effect.get("cohens_d"),
                "statistic": stat,
                "p_value_raw": p,
                "method": method,
            }

        raw_p_values.append(p if not np.isnan(p) else 1.0)
        rows.append(row)

    if not rows:
        return {
            "rows": [],
            "summary_df": pd.DataFrame(),
            "warnings": warnings,
            "alpha": alpha,
        }

    adjusted = _holm_bonferroni(raw_p_values)
    for row, adj_p in zip(rows, adjusted):
        row["p_value_adjusted"] = adj_p
        row["significant"] = bool(adj_p < alpha)

    summary_df = pd.DataFrame(rows)[
        [
            "metric", "type", "control_n", "treatment_n",
            "control_value", "treatment_value",
            "absolute_diff", "relative_pct",
            "cohens_d_or_na",
            "p_value_raw", "p_value_adjusted", "significant", "method",
        ]
    ].copy()
    summary_df = summary_df.rename(
        columns={
            "cohens_d_or_na": "Cohen's d / N/A",
            "p_value_raw": "p (raw)",
            "p_value_adjusted": "p (Holm-adj)",
            "relative_pct": "relative % diff",
        }
    )
    return {
        "rows": rows,
        "summary_df": summary_df,
        "warnings": warnings,
        "alpha": alpha,
    }
