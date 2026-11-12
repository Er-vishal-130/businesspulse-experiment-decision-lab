from __future__ import annotations

from math import ceil, isfinite
from numbers import Integral

import numpy as np
from scipy.stats import norm


def _validate_rates(*rates: float) -> tuple[float, ...]:
    try:
        values = tuple(float(rate) for rate in rates)
    except (TypeError, ValueError) as exc:
        raise ValueError("Conversion rates must be finite numbers between 0 and 1.") from exc
    if not all(isfinite(rate) and 0 <= rate <= 1 for rate in values):
        raise ValueError("Conversion rates must be finite numbers between 0 and 1.")
    return values


def _validate_design(alpha: float, power: float) -> tuple[float, float]:
    try:
        alpha_value = float(alpha)
        power_value = float(power)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alpha and desired power must be finite numbers between 0 and 1.") from exc
    if not isfinite(alpha_value) or not 0 < alpha_value < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not isfinite(power_value) or not 0 < power_value < 1:
        raise ValueError("power must be between 0 and 1.")
    return alpha_value, power_value


def required_sample_size(
    base_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int | None:
    base_rate, treatment_rate = _validate_rates(base_rate, treatment_rate)
    alpha, power = _validate_design(alpha, power)
    effect = abs(treatment_rate - base_rate)
    if effect == 0 or base_rate in (0, 1) or treatment_rate in (0, 1):
        return None
    p_bar = (base_rate + treatment_rate) / 2.0
    z_alpha = norm.ppf(1 - alpha / 2.0) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(base_rate * (1 - base_rate) + treatment_rate * (1 - treatment_rate))
    ) ** 2
    n_per_group = numerator / (effect ** 2)
    return max(int(ceil(n_per_group)), 2)


def minimum_detectable_effect(
    sample_size_per_group: int,
    base_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> float | None:
    base_rate, = _validate_rates(base_rate)
    alpha, power = _validate_design(alpha, power)
    if (
        isinstance(sample_size_per_group, bool)
        or not isinstance(sample_size_per_group, Integral)
    ):
        raise ValueError("sample_size_per_group must be an integer.")
    if sample_size_per_group < 2 or base_rate in (0, 1):
        return None
    p_bar = base_rate
    z_alpha = norm.ppf(1 - alpha / 2.0) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    numerator = z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(base_rate * (1 - base_rate) + p_bar * (1 - p_bar))
    effect = float(numerator / np.sqrt(sample_size_per_group))
    if not isfinite(effect) or effect > max(base_rate, 1 - base_rate):
        return None
    return effect
