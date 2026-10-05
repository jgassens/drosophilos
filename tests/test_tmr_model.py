"""Checks against closed forms, enumerated missions, and independent event sampling."""

from dataclasses import replace
import itertools
import json
import math

import numpy as np
import pytest

from drosophilos.bench.tmr_model import (
    BASE_STALL_TICKS,
    RATE_ROBUST_STALL_TICKS,
    Model,
    analytic,
    binned_hazard,
    campaign_failure,
    main,
    monte_carlo,
    observed_hazard,
    unreplicated_failure,
)


def enumerate_mission(hazards, recovery):
    """Independent oracle: enumerate every per-lane Bernoulli schedule, including
    draws ignored while that lane is down. Track actual lane return dates.
    """
    total = 0.0
    for flat in itertools.product((0, 1), repeat=3 * len(hazards)):
        events = np.array(flat).reshape(len(hazards), 3)
        probability = math.prod(h if event else 1 - h
                                for row, h in zip(events, hazards) for event in row)
        available_at = [0, 0, 0]
        for t, row in enumerate(events):
            for lane, event in enumerate(row):
                if t >= available_at[lane] and event:
                    available_at[lane] = len(hazards) if recovery is None else t + recovery
            if sum(date > t for date in available_at) >= 2:
                total += probability
                break
    return total


@pytest.mark.parametrize("recovery", [1, 2, 3, 100, None])
def test_analytic_matches_exhaustive_missions(recovery):
    hazards = [0.13, 0.31, 0.07]
    actual = analytic(Model(ticks=3, copies=1, hazard=hazards, recovery_ticks=recovery))
    assert actual["group_failure"] == pytest.approx(enumerate_mission(hazards, recovery), abs=2e-14)


def test_no_recovery_is_binomial_first_stall_probability():
    h = np.linspace(1e-5, 9e-5, 1000)
    p = -math.expm1(float(np.log1p(-h).sum()))
    expected = 3 * p * p - 2 * p**3
    result = analytic(Model(hazard=h, recovery_ticks=None))
    assert result["group_failure"] == pytest.approx(expected, rel=2e-12)
    assert result["campaign_failure"] == pytest.approx(-math.expm1(100 * math.log1p(-expected)))


def test_one_tick_repair_does_not_erase_same_tick_double_stalls():
    h = np.array([0.1, 0.2, 0.3, 0.01])
    majority_loss = 3 * h * h - 2 * h**3
    expected = -math.expm1(float(np.log1p(-majority_loss).sum()))
    result = analytic(Model(ticks=4, copies=7, hazard=h, recovery_ticks=1))
    assert result["group_failure"] == pytest.approx(expected)
    assert result["campaign_failure"] == pytest.approx(1 - (1 - expected) ** 7)


def test_long_repair_and_failed_repair_reduce_to_no_recovery():
    base = Model(ticks=12, hazard=0.05, recovery_ticks=None)
    expected = analytic(base)["group_failure"]
    for r in (12, 13, 100000):
        assert analytic(replace(base, recovery_ticks=r))["group_failure"] == pytest.approx(expected)
    assert analytic(replace(base, recovery_ticks=1, recovery_failure=1))["group_failure"] == pytest.approx(expected)
    assert analytic(replace(base, recovery_ticks=3, recovery_failure=1))["group_failure"] == pytest.approx(expected)


def test_recovery_duration_failure_and_exposure_increase_risk():
    model = Model(ticks=50, copies=1, hazard=0.01)
    risks = [analytic(replace(model, recovery_ticks=r))["group_failure"] for r in (1, 2, 10, None)]
    assert risks == sorted(risks) and len(set(risks)) == 4
    assert analytic(replace(model, recovery_failure=0.2))["group_failure"] > analytic(model)["group_failure"]
    curve = analytic(model)["group_failure_by_tick"]
    assert np.all(np.diff(curve) >= 0)
    assert curve[-1] == analytic(model)["group_failure"]


def test_protection_failure_and_global_correlation_are_separate():
    base = Model(ticks=12, copies=9, hazard=0.02, recovery_ticks=3)
    local = replace(base, voter_failure=0.01, watchdog_failure=0.02, common_mode=0.03)
    expected_group = 1 - (1 - analytic(base)["group_failure"]) * (0.99 * 0.98 * 0.97) ** 12
    assert analytic(local)["group_failure"] == pytest.approx(expected_group)
    correlated = replace(local, global_common_mode=0.04)
    expected_campaign = 1 - (1 - expected_group) ** 9 * 0.96**12
    assert analytic(correlated)["campaign_failure"] == pytest.approx(expected_campaign)
    # Shared shocks do not get 9 independent opportunities per tick.
    for copies in (1, 100):
        only_global = Model(ticks=12, copies=copies, hazard=0, global_common_mode=0.04)
        assert analytic(only_global)["campaign_failure"] == pytest.approx(1 - 0.96**12)


def test_tiny_probabilities_survive_floating_point_subtraction():
    result = analytic(Model(hazard=1e-12, recovery_ticks=1))
    assert result["group_failure"] == pytest.approx(3e-21, rel=1e-8, abs=0)
    assert result["campaign_failure"] == pytest.approx(3e-19, rel=1e-8, abs=0)
    assert campaign_failure(1e-20, 100) == pytest.approx(1e-18, rel=1e-12, abs=0)


def test_measured_scale_is_not_a_zero_failure_prediction():
    assert unreplicated_failure() == pytest.approx(0.9932628952, abs=1e-9)
    assert 0.49 < analytic(Model(recovery_ticks=None))["campaign_failure"] < 0.51
    assert 0.002 < analytic(Model(recovery_ticks=2))["campaign_failure"] < 0.003


@pytest.mark.parametrize("recovery", [1, 2, 4, None])
def test_monte_carlo_matches_analytic_for_nonstationary_campaigns(recovery):
    model = Model(ticks=5, copies=3, hazard=[0.01, 0.13, 0, 0.06, 0.08],
                  recovery_ticks=recovery, recovery_failure=0.15,
                  voter_failure=0.001, watchdog_failure=0.002,
                  common_mode=0.003, global_common_mode=0.005)
    exact = analytic(model)["campaign_failure"]
    mc = monte_carlo(model, trials=25000, seed=38, batch_size=700)
    sigma = math.sqrt(exact * (1 - exact) / mc["trials"])
    assert abs(mc["campaign_failure"] - exact) < 6 * sigma
    assert mc["confidence_95"][0] <= mc["campaign_failure"] <= mc["confidence_95"][1]
    assert mc["failures"] / mc["trials"] == mc["campaign_failure"]


def test_monte_carlo_repeats_replica_failures_after_repair():
    model = Model(ticks=15, copies=1, hazard=0.15, recovery_ticks=1)
    exact = analytic(model)["campaign_failure"]
    sampled = monte_carlo(model, trials=30000, seed=59)["campaign_failure"]
    assert abs(sampled - exact) < 0.012
    assert sampled < analytic(replace(model, recovery_ticks=None))["campaign_failure"] - 0.2


@pytest.mark.parametrize("kwargs", [
    {"hazard": 1}, {"voter_failure": 1}, {"watchdog_failure": 1},
    {"common_mode": [0, 1, 0]}, {"global_common_mode": [0, 0, 1]},
])
def test_certain_failure_endpoints(kwargs):
    settings = {"ticks": 3, "copies": 4, "hazard": 0.0, **kwargs}
    model = Model(**settings)
    assert analytic(model)["campaign_failure"] == 1
    mc = monte_carlo(model, trials=17, batch_size=5)
    assert mc["failures"] == 17


def test_zero_risk_and_sampling_reproducibility():
    model = Model(ticks=5, copies=2, hazard=0)
    assert analytic(model)["campaign_failure"] == 0
    result = monte_carlo(model, trials=100, seed=7)
    assert result["failures"] == 0
    assert result["confidence_95"][1] > 0  # zero observations are not proof of zero risk
    noisy = replace(model, hazard=0.05)
    assert monte_carlo(noisy, trials=100, seed=7) == monte_carlo(noisy, trials=100, seed=7)


def test_empirical_exposure_censors_at_first_failed_attempt():
    # One replica fails at attempt 0; the other supplies 4 attempted ticks.
    h = binned_hazard([0], copies=2, ticks=4, edges=[0, 2, 4])
    np.testing.assert_allclose(h, [1 / 3, 1 / 3, 0, 0])
    with pytest.raises(ValueError, match="no at-risk"):
        binned_hazard([0], copies=1, ticks=4, edges=[0, 2, 4])


def test_observed_profile_preserves_event_indices_and_mission_risk():
    assert BASE_STALL_TICKS == (1, 31, 323, 728)
    assert RATE_ROBUST_STALL_TICKS == (450, 535, 599, 654, 864)
    raw = observed_hazard(None)
    assert raw[500] > raw[250] > raw[0] > raw[750] > 0
    normalized = observed_hazard()
    assert unreplicated_failure(normalized) == pytest.approx(unreplicated_failure(), rel=2e-13)
    a = analytic(Model(hazard=normalized, recovery_ticks=None))["campaign_failure"]
    b = analytic(Model(recovery_ticks=None))["campaign_failure"]
    assert a == pytest.approx(b, rel=2e-12)
    # Clustering changes the chance of overlapping recovery windows.
    assert analytic(Model(hazard=normalized))["campaign_failure"] > analytic(Model())["campaign_failure"]


@pytest.mark.parametrize("kwargs", [
    {"ticks": 0}, {"ticks": 1.5}, {"copies": -1}, {"copies": True},
    {"recovery_ticks": 0}, {"recovery_ticks": 0.5}, {"recovery_failure": -0.1},
    {"recovery_failure": float("nan")}, {"hazard": -1}, {"hazard": float("inf")},
    {"hazard": [0.1]}, {"hazard": [[0.1] * 1000]}, {"voter_failure": 1.1},
    {"watchdog_failure": float("nan")}, {"global_common_mode": -0.1},
])
def test_invalid_model_parameters(kwargs):
    with pytest.raises(ValueError):
        Model(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"edges": [0, 2, 2, 4]}, {"edges": [1, 4]}, {"stall_ticks": [4]},
    {"stall_ticks": [0, 1, 2]}, {"stall_ticks": [0.5]}, {"normalize_to": 1},
    {"stall_ticks": [], "normalize_to": 0.01},
])
def test_invalid_empirical_parameters(kwargs):
    params = dict(stall_ticks=[0], copies=2, ticks=4, edges=[0, 4])
    with pytest.raises(ValueError):
        binned_hazard(**{**params, **kwargs})


def test_cli_report_and_validation(capsys):
    main(["--ticks", "4", "--copies", "2", "--trials", "100", "--hazard", "0.1",
          "--recovery-ticks", "none"])
    report = json.loads(capsys.readouterr().out)
    assert report["parameters"]["recovery_ticks"] == "none"
    assert report["monte_carlo"]["trials"] == 100
    assert report["analytic"]["campaign_failure"] < report["unreplicated_stalls_only"]
    with pytest.raises(SystemExit):
        main(["--profile", "observed", "--ticks", "5"])
    with pytest.raises(ValueError):
        monte_carlo(Model(), trials=0)
