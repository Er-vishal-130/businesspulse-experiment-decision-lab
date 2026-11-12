"""Bayesian A/B testing using Beta-Binomial conjugate model.

Provides posterior analysis, credible intervals, probability of being best,
expected loss, and Plotly charts — all without external AI services.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import beta as beta_dist


def _extract_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Return (successes, trials) from a group DataFrame."""
    values = pd.to_numeric(df.get("converted", pd.Series(dtype=float)), errors="coerce")
    values = values[values.isin([0, 1])]
    return int(values.sum()), int(len(values))


def run_bayesian_ab_test(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_mass: float = 0.95,
    n_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Bayesian posterior summaries for a binary conversion experiment.

    Parameters
    ----------
    control_df, treatment_df:
        DataFrames with a binary ``converted`` column (0/1).
    prior_alpha, prior_beta:
        Beta prior parameters (default: Beta(1,1) = uniform / uninformative).
    credible_mass:
        Width of the highest-density credible interval (e.g. 0.95).
    n_samples:
        Monte-Carlo sample count for probability and expected-loss estimates.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with posterior statistics, credible intervals, and Monte-Carlo estimates.
    """
    if not (0 < credible_mass < 1):
        raise ValueError("credible_mass must be between 0 and 1.")
    if not (prior_alpha > 0 and prior_beta > 0):
        raise ValueError("Prior parameters must be positive.")

    control_s, control_n = _extract_counts(control_df)
    treatment_s, treatment_n = _extract_counts(treatment_df)

    warnings: list[str] = []
    if control_n == 0 or treatment_n == 0:
        return {
            "control_successes": control_s,
            "control_trials": control_n,
            "treatment_successes": treatment_s,
            "treatment_trials": treatment_n,
            "error": "At least one group has no valid binary observations.",
            "warnings": warnings,
        }
    if min(control_n, treatment_n) < 30:
        warnings.append(
            "Fewer than 30 valid observations in at least one group; "
            "posterior estimates are highly sensitive to the prior."
        )

    # Posterior parameters: Beta(alpha + successes, beta + failures)
    ctrl_a = prior_alpha + control_s
    ctrl_b = prior_beta + (control_n - control_s)
    trt_a = prior_alpha + treatment_s
    trt_b = prior_beta + (treatment_n - treatment_s)

    ctrl_mean = ctrl_a / (ctrl_a + ctrl_b)
    trt_mean = trt_a / (trt_a + trt_b)

    half_tail = (1 - credible_mass) / 2
    ctrl_ci_low, ctrl_ci_high = beta_dist.ppf([half_tail, 1 - half_tail], ctrl_a, ctrl_b)
    trt_ci_low, trt_ci_high = beta_dist.ppf([half_tail, 1 - half_tail], trt_a, trt_b)

    # Monte-Carlo estimates
    rng = np.random.default_rng(seed)
    ctrl_samples = rng.beta(ctrl_a, ctrl_b, n_samples)
    trt_samples = rng.beta(trt_a, trt_b, n_samples)
    prob_treatment_best = float((trt_samples > ctrl_samples).mean())

    # Expected loss: average shortfall when choosing the inferior option
    loss_control = float(np.maximum(trt_samples - ctrl_samples, 0).mean())
    loss_treatment = float(np.maximum(ctrl_samples - trt_samples, 0).mean())

    diff_samples = trt_samples - ctrl_samples
    diff_ci_low = float(np.percentile(diff_samples, half_tail * 100))
    diff_ci_high = float(np.percentile(diff_samples, (1 - half_tail) * 100))
    diff_mean = float(diff_samples.mean())

    return {
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "control_successes": control_s,
        "control_trials": control_n,
        "treatment_successes": treatment_s,
        "treatment_trials": treatment_n,
        # Posterior parameters
        "ctrl_posterior_alpha": ctrl_a,
        "ctrl_posterior_beta": ctrl_b,
        "trt_posterior_alpha": trt_a,
        "trt_posterior_beta": trt_b,
        # Posterior means
        "control_posterior_mean": ctrl_mean,
        "treatment_posterior_mean": trt_mean,
        # Credible intervals
        "credible_mass": credible_mass,
        "control_credible_low": float(ctrl_ci_low),
        "control_credible_high": float(ctrl_ci_high),
        "treatment_credible_low": float(trt_ci_low),
        "treatment_credible_high": float(trt_ci_high),
        # Difference in conversion rates
        "difference_mean": diff_mean,
        "difference_credible_low": diff_ci_low,
        "difference_credible_high": diff_ci_high,
        # Monte-Carlo summaries
        "prob_treatment_best": prob_treatment_best,
        "expected_loss_if_choose_control": loss_control,
        "expected_loss_if_choose_treatment": loss_treatment,
        "n_mc_samples": n_samples,
        "warnings": warnings,
    }


def posterior_density_chart(result: dict[str, Any], n_points: int = 400) -> go.Figure:
    """Return a Plotly figure showing the Beta posterior PDFs for both groups."""
    if "error" in result:
        fig = go.Figure()
        fig.update_layout(title="Bayesian posterior — insufficient data")
        return fig

    ctrl_a, ctrl_b = result["ctrl_posterior_alpha"], result["ctrl_posterior_beta"]
    trt_a, trt_b = result["trt_posterior_alpha"], result["trt_posterior_beta"]

    x = np.linspace(0, 1, n_points)
    ctrl_pdf = beta_dist.pdf(x, ctrl_a, ctrl_b)
    trt_pdf = beta_dist.pdf(x, trt_a, trt_b)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=ctrl_pdf, mode="lines", name="Control posterior",
            line=dict(color="#64748b", width=2),
            fill="tozeroy", fillcolor="rgba(100,116,139,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=trt_pdf, mode="lines", name="Treatment posterior",
            line=dict(color="#5c7cfa", width=2),
            fill="tozeroy", fillcolor="rgba(92,124,250,0.15)",
        )
    )
    # Mark posterior means
    ctrl_mean = result["control_posterior_mean"]
    trt_mean = result["treatment_posterior_mean"]
    fig.add_vline(x=ctrl_mean, line_dash="dot", line_color="#94a3b8",
                  annotation_text=f"Control {ctrl_mean:.3%}", annotation_position="top left")
    fig.add_vline(x=trt_mean, line_dash="dot", line_color="#818cf8",
                  annotation_text=f"Treatment {trt_mean:.3%}", annotation_position="top right")

    credible_mass = result.get("credible_mass", 0.95)
    fig.update_layout(
        title=f"Bayesian posterior distributions ({credible_mass:.0%} Beta posteriors)",
        xaxis_title="Conversion rate",
        yaxis_title="Posterior density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )
    return fig


def credible_interval_chart(result: dict[str, Any]) -> go.Figure:
    """Return a Plotly figure comparing posterior means and credible intervals."""
    if "error" in result:
        fig = go.Figure()
        fig.update_layout(title="Credible intervals — insufficient data")
        return fig

    groups = ["Control", "Treatment"]
    means = [result["control_posterior_mean"], result["treatment_posterior_mean"]]
    lows = [result["control_credible_low"], result["treatment_credible_low"]]
    highs = [result["control_credible_high"], result["treatment_credible_high"]]
    error_low = [m - lo for m, lo in zip(means, lows)]
    error_high = [hi - m for m, hi in zip(means, highs)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_high,
                arrayminus=error_low,
                color="#5c7cfa",
                thickness=2,
                width=8,
            ),
            marker_color=["#64748b", "#5c7cfa"],
            text=[f"{m:.3%}" for m in means],
            textposition="outside",
            name="Posterior mean",
        )
    )
    credible_mass = result.get("credible_mass", 0.95)
    fig.update_layout(
        title=f"Posterior means with {credible_mass:.0%} credible intervals",
        yaxis_title="Conversion rate",
        yaxis_tickformat=".1%",
        margin=dict(t=60),
    )
    return fig
