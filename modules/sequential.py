"""Sequential / interim analysis for A/B experiments.

Provides:
- Sequential Probability Ratio Test (SPRT) for binary outcomes.
- O'Brien-Fleming alpha-spending boundary approximation for a fixed
  number of planned looks.
- A "safe to stop?" helper that combines SPRT and OBF signals.

All calculations assume independent binary conversion observations and
are approximate normal-approximation or exact SPRT methods.  They are
not a substitute for a pre-registered sequential design.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# ---------------------------------------------------------------------------
# SPRT
# ---------------------------------------------------------------------------

def run_sprt(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    h0_delta: float = 0.0,
    h1_delta: float | None = None,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> dict[str, Any]:
    """Sequential Probability Ratio Test for binary conversion outcomes.

    Under H0: Treatment rate = Control rate + h0_delta (default 0).
    Under H1: Treatment rate = Control rate + h1_delta.

    If h1_delta is None, it defaults to a "small" practical effect of
    max(0.01, 0.5 * observed_absolute_lift).

    The SPRT thresholds are:
        Upper boundary A = log((1 - beta) / alpha)   → reject H0 (effect detected)
        Lower boundary B = log(beta / (1 - alpha))   → accept H0 (no meaningful effect)

    Returns:
        log_lambda   : current log likelihood ratio
        decision     : "Reject H0 (effect detected)", "Accept H0 (no effect)",
                       or "Continue collecting data"
        boundary_upper / boundary_lower: the boundaries in log-lambda space
        safe_to_stop : bool
        warnings     : list[str]
    """
    warnings_list: list[str] = [
        "SPRT assumes independent binary observations and a pre-specified "
        "alternative rate.  Repeated observation or post-hoc selection of h1_delta "
        "can inflate false-positive rates."
    ]

    ctrl_vals = pd.to_numeric(
        control_df.get("converted", pd.Series(dtype=float)), errors="coerce"
    )
    trt_vals = pd.to_numeric(
        treatment_df.get("converted", pd.Series(dtype=float)), errors="coerce"
    )
    ctrl_vals = ctrl_vals[ctrl_vals.isin([0, 1])]
    trt_vals = trt_vals[trt_vals.isin([0, 1])]

    ctrl_n, trt_n = int(len(ctrl_vals)), int(len(trt_vals))
    ctrl_s, trt_s = int(ctrl_vals.sum()), int(trt_vals.sum())

    if ctrl_n == 0 or trt_n == 0:
        return {
            "log_lambda": float("nan"),
            "decision": "Not estimable",
            "boundary_upper": float("nan"),
            "boundary_lower": float("nan"),
            "safe_to_stop": False,
            "warnings": warnings_list + ["Both groups need at least one observation."],
        }

    p0 = ctrl_s / ctrl_n  # H0 control rate
    observed_lift = trt_s / trt_n - p0

    if h1_delta is None:
        h1_delta = max(0.01, abs(observed_lift) * 0.5)
        warnings_list.append(
            f"h1_delta was not pre-specified; defaulted to {h1_delta:.4f} "
            "(half the observed absolute lift). Pre-register the alternative before data collection."
        )

    p1_ctrl = p0 + h0_delta
    p1_trt = p0 + h1_delta

    # Clamp to valid probability range
    p1_ctrl = max(1e-9, min(1 - 1e-9, p1_ctrl))
    p1_trt = max(1e-9, min(1 - 1e-9, p1_trt))

    boundary_upper = math.log((1 - beta) / alpha)
    boundary_lower = math.log(beta / (1 - alpha))

    # Compute log-likelihood ratio for Treatment arm only
    # LLR = Σ [ x_i * log(p1/p0) + (1-x_i) * log((1-p1)/(1-p0)) ]
    if abs(p1_trt - p1_ctrl) < 1e-12:
        log_lambda = 0.0
    else:
        log_p_ratio = math.log(p1_trt / p1_ctrl) if p1_ctrl > 0 else float("nan")
        log_1mp_ratio = math.log((1 - p1_trt) / (1 - p1_ctrl)) if (1 - p1_ctrl) > 0 else float("nan")
        if math.isnan(log_p_ratio) or math.isnan(log_1mp_ratio):
            log_lambda = float("nan")
        else:
            log_lambda = float(trt_s * log_p_ratio + (trt_n - trt_s) * log_1mp_ratio)

    if math.isnan(log_lambda):
        decision = "Not estimable"
        safe_to_stop = False
    elif log_lambda >= boundary_upper:
        decision = "Reject H0 — effect detected; safe to stop."
        safe_to_stop = True
    elif log_lambda <= boundary_lower:
        decision = "Accept H0 — no meaningful effect detected; safe to stop."
        safe_to_stop = True
    else:
        decision = "Continue collecting data — test has not crossed either boundary."
        safe_to_stop = False

    return {
        "log_lambda": log_lambda,
        "decision": decision,
        "boundary_upper": boundary_upper,
        "boundary_lower": boundary_lower,
        "safe_to_stop": safe_to_stop,
        "h0_delta": h0_delta,
        "h1_delta": h1_delta,
        "alpha": alpha,
        "beta": beta,
        "control_successes": ctrl_s,
        "control_trials": ctrl_n,
        "treatment_successes": trt_s,
        "treatment_trials": trt_n,
        "warnings": warnings_list,
    }


# ---------------------------------------------------------------------------
# O'Brien-Fleming alpha spending
# ---------------------------------------------------------------------------

def obf_spending_boundary(
    info_fraction: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Compute the O'Brien-Fleming alpha-spending boundary at a given information fraction.

    Uses the Lan-DeMets approximation:
        alpha_spent(t) ≈ 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))   [two-sided]

    Parameters
    ----------
    info_fraction:
        Current information fraction t ∈ (0, 1].  Typically n_now / n_planned.
    alpha:
        Overall significance level.
    two_sided:
        If True, compute two-sided boundaries.

    Returns
    -------
    z_boundary: the z-score boundary at this interim look.
    """
    if not (0 < info_fraction <= 1):
        raise ValueError("info_fraction must be in (0, 1].")
    z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
    # OBF Lan-DeMets spending: alpha_spent(t) = 2*(1 - Phi(z_alpha/sqrt(t)))
    boundary_z = z_alpha / math.sqrt(info_fraction)
    return float(boundary_z)


def interim_analysis(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    planned_n_per_group: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> dict[str, Any]:
    """O'Brien-Fleming interim analysis for binary conversions.

    Computes the current z-statistic and the OBF boundary at the current
    information fraction, and reports whether the boundary is crossed.
    """
    warnings_list: list[str] = [
        "O'Brien-Fleming boundaries are approximate (Lan-DeMets).  The planned "
        "sample size must be pre-registered; post-hoc specification inflates type-I error."
    ]
    ctrl_vals = pd.to_numeric(
        control_df.get("converted", pd.Series(dtype=float)), errors="coerce"
    )
    trt_vals = pd.to_numeric(
        treatment_df.get("converted", pd.Series(dtype=float)), errors="coerce"
    )
    ctrl_vals = ctrl_vals[ctrl_vals.isin([0, 1])]
    trt_vals = trt_vals[trt_vals.isin([0, 1])]

    ctrl_n, trt_n = int(len(ctrl_vals)), int(len(trt_vals))
    ctrl_s, trt_s = int(ctrl_vals.sum()), int(trt_vals.sum())

    if ctrl_n == 0 or trt_n == 0 or planned_n_per_group <= 0:
        return {
            "z_statistic": float("nan"),
            "obf_boundary": float("nan"),
            "information_fraction": float("nan"),
            "decision": "Not estimable",
            "safe_to_stop": False,
            "warnings": warnings_list,
        }

    current_n = (ctrl_n + trt_n) / 2
    info_fraction = min(current_n / planned_n_per_group, 1.0)

    p_ctrl = ctrl_s / ctrl_n
    p_trt = trt_s / trt_n
    pooled_p = (ctrl_s + trt_s) / (ctrl_n + trt_n)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / ctrl_n + 1 / trt_n))

    if se < 1e-12:
        z_stat = 0.0
    else:
        z_stat = (p_trt - p_ctrl) / se

    try:
        obf_z = obf_spending_boundary(max(info_fraction, 1e-6), alpha=alpha, two_sided=two_sided)
    except ValueError:
        obf_z = float("nan")

    crossed = abs(z_stat) >= obf_z if not math.isnan(obf_z) else False
    if math.isnan(obf_z):
        decision = "Boundary could not be computed."
    elif crossed:
        direction = "higher" if z_stat > 0 else "lower"
        decision = (
            f"OBF boundary crossed — the current z-statistic ({z_stat:.3f}) exceeds "
            f"the boundary ({obf_z:.3f}). Treatment conversion appears {direction}. "
            "Safe to stop early."
        )
    else:
        decision = (
            f"OBF boundary not yet crossed (z={z_stat:.3f}, boundary={obf_z:.3f}). "
            f"Information fraction: {info_fraction:.1%}. Continue collecting data."
        )

    if info_fraction < 0.3:
        warnings_list.append(
            "Very low information fraction (<30%). Early stopping is unreliable; "
            "boundary crossing here may be a random fluctuation."
        )

    return {
        "z_statistic": float(z_stat),
        "obf_boundary": float(obf_z) if not math.isnan(obf_z) else float("nan"),
        "information_fraction": float(info_fraction),
        "decision": decision,
        "safe_to_stop": crossed,
        "control_successes": ctrl_s,
        "control_trials": ctrl_n,
        "treatment_successes": trt_s,
        "treatment_trials": trt_n,
        "planned_n_per_group": planned_n_per_group,
        "alpha": alpha,
        "two_sided": two_sided,
        "warnings": warnings_list,
    }


def sprt_boundary_chart(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    sprt_result: dict[str, Any],
) -> go.Figure:
    """Visualise SPRT log-LR trajectory vs. boundaries."""
    upper = sprt_result.get("boundary_upper", float("nan"))
    lower = sprt_result.get("boundary_lower", float("nan"))
    ll = sprt_result.get("log_lambda", float("nan"))

    trt_vals = pd.to_numeric(
        treatment_df.get("converted", pd.Series(dtype=float)), errors="coerce"
    )
    trt_vals = trt_vals[trt_vals.isin([0, 1])].reset_index(drop=True)

    p0 = sprt_result.get("control_successes", 0) / max(sprt_result.get("control_trials", 1), 1)
    p1 = p0 + sprt_result.get("h1_delta", 0.01)
    p0 = max(1e-9, min(1 - 1e-9, p0))
    p1 = max(1e-9, min(1 - 1e-9, p1))

    log_p_ratio = math.log(p1 / p0) if p0 > 0 else 0.0
    log_1mp_ratio = math.log((1 - p1) / (1 - p0)) if (1 - p0) > 0 else 0.0

    cumulative_ll = [0.0]
    running = 0.0
    for obs in trt_vals:
        running += obs * log_p_ratio + (1 - obs) * log_1mp_ratio
        cumulative_ll.append(running)

    x = list(range(len(cumulative_ll)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=cumulative_ll, mode="lines", name="Log LR",
                   line=dict(color="#5c7cfa", width=2))
    )
    if not math.isnan(upper):
        fig.add_hline(y=upper, line_dash="dash", line_color="#22c55e",
                      annotation_text=f"Upper boundary ({upper:.2f})",
                      annotation_position="bottom right")
    if not math.isnan(lower):
        fig.add_hline(y=lower, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Lower boundary ({lower:.2f})",
                      annotation_position="top right")
    fig.update_layout(
        title="SPRT log-likelihood ratio trajectory",
        xaxis_title="Observations (Treatment arm)",
        yaxis_title="Cumulative log-LR",
        margin=dict(t=60),
    )
    return fig
