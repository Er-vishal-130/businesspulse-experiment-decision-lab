import math

import pytest
from scipy.stats import norm

from modules.power_analysis import (
    minimum_detectable_effect,
    required_sample_size,
)


def test_required_sample_size_matches_independent_two_proportion_approximation():
    base_rate = 0.10
    treatment_rate = 0.12
    alpha = 0.05
    desired_power = 0.80
    pooled = (base_rate + treatment_rate) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(desired_power)
    expected = math.ceil(
        (
            z_alpha * math.sqrt(2 * pooled * (1 - pooled))
            + z_beta
            * math.sqrt(
                base_rate * (1 - base_rate)
                + treatment_rate * (1 - treatment_rate)
            )
        )
        ** 2
        / (treatment_rate - base_rate) ** 2
    )

    assert required_sample_size(base_rate, treatment_rate) == expected


def test_zero_effect_and_boundary_rates_are_unavailable_not_misreported():
    assert required_sample_size(0.1, 0.1) is None
    assert required_sample_size(0.0, 0.1) is None
    assert required_sample_size(0.9, 1.0) is None
    assert minimum_detectable_effect(100, 0.0) is None
    assert minimum_detectable_effect(1, 0.5) is None
    assert minimum_detectable_effect(2, 0.5) is None


def test_mde_matches_equal_allocation_normal_approximation():
    sample_size = 1000
    base_rate = 0.1
    z_alpha = norm.ppf(1 - 0.05 / 2)
    z_beta = norm.ppf(0.8)
    expected = (
        z_alpha * math.sqrt(2 * base_rate * (1 - base_rate))
        + z_beta * math.sqrt(2 * base_rate * (1 - base_rate))
    ) / math.sqrt(sample_size)

    assert minimum_detectable_effect(sample_size, base_rate) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("base_rate", "treatment_rate"),
    [
        (-0.1, 0.2),
        (0.2, 1.1),
        (float("nan"), 0.2),
    ],
)
def test_invalid_rates_are_rejected(base_rate, treatment_rate):
    with pytest.raises(ValueError):
        required_sample_size(base_rate, treatment_rate)


@pytest.mark.parametrize(
    ("alpha", "power"),
    [
        (0.0, 0.8),
        (1.0, 0.8),
        (0.05, 0.0),
        (0.05, 1.0),
        (float("nan"), 0.8),
    ],
)
def test_invalid_design_inputs_are_rejected(alpha, power):
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.2, alpha=alpha, power=power)
    with pytest.raises(ValueError):
        minimum_detectable_effect(100, 0.1, alpha=alpha, power=power)


def test_mde_rejects_non_integer_sample_size():
    with pytest.raises(ValueError):
        minimum_detectable_effect(10.5, 0.2)
