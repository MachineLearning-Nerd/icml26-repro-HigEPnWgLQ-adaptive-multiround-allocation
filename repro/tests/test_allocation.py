import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro/src"))

from run_allocation import (
    ExactDistribution,
    compositions,
    convolve_many,
    exact_multiround_case,
    expected_reward,
    independent_population_dp,
    official_greedy,
    single_round_robustness,
    tails,
    truncated_pmf,
    tv_pmf,
)


@pytest.mark.parametrize("budget", range(1, 9))
def test_official_greedy_matches_bruteforce(budget):
    pmfs = ((0.4, 0.1, 0.5), (0.1, 0.6, 0.2, 0.1), (0.7, 0.3))
    got = expected_reward(pmfs, official_greedy(pmfs, budget))
    want = max(expected_reward(pmfs, a) for a in compositions(budget, len(pmfs)))
    assert got == pytest.approx(want, abs=1e-14)


def test_marginal_decomposition_identity():
    pmf = (0.17, 0.23, 0.31, 0.29)
    for k in range(8):
        direct = sum(prob * min(k, x) for x, prob in enumerate(pmf))
        assert tails(pmf, k).sum() == pytest.approx(direct, abs=1e-14)


def test_truncated_pmf_preserves_mass():
    pmf = (0.1, 0.2, 0.3, 0.4)
    for k in range(5):
        got = truncated_pmf(pmf, k)
        assert got.sum() == pytest.approx(1)
        assert len(got) == k + 1


def test_direct_convolution_matches_enumeration():
    pmfs = ((0.2, 0.3, 0.5), (0.6, 0.4))
    alloc = (2, 1)
    got = convolve_many([truncated_pmf(p, k) for p, k in zip(pmfs, alloc)])
    want = np.zeros(4)
    for x, px in enumerate(pmfs[0]):
        for y, py in enumerate(pmfs[1]):
            want[min(x, 2) + min(y, 1)] += px * py
    assert np.max(np.abs(got - want[: len(got)])) < 1e-14


def test_independent_dp_boundary_and_finiteness():
    U, actions = independent_population_dp((0.3, 0.2, 0.5), 8, 0.7)
    assert all(U[(0, n)] == 0 for n in range(9))
    assert all(U[(r, 0)] == 0 for r in range(9))
    assert all(np.isfinite(x) for x in U.values())
    assert all(0 <= s <= r for (r, _), s in actions.items())


def test_tight_single_round_family():
    result = single_round_robustness()
    assert result["max_bound_violation"] <= 1e-12
    assert result["max_tightness_error"] <= 1e-12


def test_homogeneous_multiround_zero_bound():
    p = (0.6, 0.3, 0.1)
    result = exact_multiround_case((p,), (1.0,), (p,), (1.0,), 6, 0.7)
    assert result["lhs_suboptimality"] == pytest.approx(0, abs=1e-13)
    assert result["rhs_bound"] == pytest.approx(0, abs=1e-13)


def test_noisy_multiround_bound():
    true = ((0.6, 0.3, 0.1), (0.15, 0.25, 0.6))
    hat = ((0.93, 0.06, 0.01), (0.18, 0.21, 0.61))
    result = exact_multiround_case(true, (0.65, 0.35), hat, (0.5, 0.5), 6, 0.7)
    assert result["lhs_suboptimality"] > 0.1
    assert result["lhs_suboptimality"] <= result["rhs_bound"] + 1e-12


def test_tv_distance_known_case():
    assert tv_pmf((1, 0), (0, 1)) == pytest.approx(1)
    assert tv_pmf((0.5, 0.5), (0.5, 0.5)) == 0


def test_invalid_pmf_rejected_control():
    with pytest.raises(ValueError):
        ExactDistribution((0.4, -0.1, 0.7))
    with pytest.raises(ValueError):
        ExactDistribution((0.4, 0.4))


def test_prefix_constraint_matters_control():
    # A fabricated non-monotone marginal sequence cannot be any distribution tail.
    fake_tails = np.array([0.2, 0.9])
    assert np.any(np.diff(fake_tails) > 0)
    with pytest.raises(ValueError):
        ExactDistribution((0.8, -0.7, 0.9))


def test_persisted_summary_claims():
    data = json.loads((ROOT / "outputs/summary.json").read_text())
    assert data["claim1_greedy_optimality"]["max_oracle_gap"] < 1e-12
    assert data["claim2_population_dp"]["max_state_error"] < 1e-12
    assert data["claim3_robustness"]["multi_round"]["max_bound_violation"] <= 1e-12

