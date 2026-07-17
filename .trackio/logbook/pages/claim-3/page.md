# Claim 3


---
<!-- trackio-cell
{"type": "code", "id": "cell_9e18a33612ba", "created_at": "2026-07-17T04:36:18+00:00", "title": "Tight robustness and exact multi-round decomposition", "command": ["python", "repro/src/run_allocation.py", "--config", "repro/configs/full.json", "--output-dir", "outputs"], "exit_code": 0, "duration_s": 9.008}
-->
````bash
$ python repro/src/run_allocation.py --config repro/configs/full.json --output-dir outputs
````

exit 0 · 9.0s


````python title=run_allocation.py
#!/usr/bin/env python3
"""Exact reproduction for arXiv:2605.12111 / OpenReview HigEPnWgLQ."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "official"))

from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from policies.our_policy import OurPolicy, precompute_surrogate


class ExactDistribution(AbstractArrivalDistribution):
    def __init__(self, pmf: tuple[float, ...], seed: int = 0):
        super().__init__(seed)
        arr = np.asarray(pmf, dtype=float)
        if arr.ndim != 1 or len(arr) == 0 or np.min(arr) < 0 or not np.isclose(arr.sum(), 1):
            raise ValueError("pmf must be a nonnegative probability vector")
        self.pdf = arr
        self.cdf = np.cumsum(arr)

    def sample(self) -> int:
        return int(np.searchsorted(self.cdf, self.rng.random()))

    def prob_equal(self, k: int) -> float:
        return float(self.pdf[k]) if 0 <= k < len(self.pdf) else 0.0

    def prob_at_least(self, k: int) -> float:
        if k <= 0:
            return 1.0
        return float(self.pdf[k:].sum()) if k < len(self.pdf) else 0.0


class CyclingPopulation(AbstractPopulation):
    """Returns an exact finite mixture whenever sample count is a multiple of its size."""

    def __init__(self, pmfs: tuple[tuple[float, ...], ...]):
        super().__init__(0)
        self.pmfs = pmfs

    def sample_arrival_distributions(self, n: int):
        return [ExactDistribution(self.pmfs[i % len(self.pmfs)], i) for i in range(n)]


def pad(pmf: tuple[float, ...], length: int) -> np.ndarray:
    out = np.zeros(length)
    out[: len(pmf)] = pmf
    return out


def tails(pmf: tuple[float, ...], budget: int) -> np.ndarray:
    a = pad(pmf, max(len(pmf), budget + 1))
    return np.array([a[k:].sum() for k in range(1, budget + 1)])


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def expected_reward(pmfs, allocation) -> float:
    return float(sum(tails(p, k).sum() for p, k in zip(pmfs, allocation)))


def official_greedy(pmfs, budget: int) -> tuple[int, ...]:
    dists = [ExactDistribution(tuple(p), i) for i, p in enumerate(pmfs)]
    return tuple(OurPolicy.greedy_single_stage(None, dists, budget))


def exhaustive_greedy_check(max_budget: int):
    library = (
        (1.0,),
        (0.7, 0.3),
        (0.2, 0.8),
        (0.1, 0.3, 0.6),
        (0.4, 0.1, 0.2, 0.3),
        (0.05, 0.15, 0.25, 0.2, 0.35),
    )
    cases = 0
    allocations = 0
    max_gap = 0.0
    max_identity_error = 0.0
    for n in range(1, 5):
        for ids in itertools.combinations_with_replacement(range(len(library)), n):
            pmfs = tuple(library[i] for i in ids)
            for budget in range(max_budget + 1):
                greedy = official_greedy(pmfs, budget)
                greedy_value = expected_reward(pmfs, greedy)
                oracle = -np.inf
                for alloc in compositions(budget, n):
                    allocations += 1
                    direct = expected_reward(pmfs, alloc)
                    # Independent E[min(k,X)] identity by PMF enumeration.
                    enum = sum(
                        sum(prob * min(k, x) for x, prob in enumerate(p))
                        for p, k in zip(pmfs, alloc)
                    )
                    max_identity_error = max(max_identity_error, abs(direct - enum))
                    oracle = max(oracle, direct)
                max_gap = max(max_gap, oracle - greedy_value)
                cases += 1
    return {
        "frontier_budget_cases": cases,
        "allocations_enumerated": allocations,
        "max_oracle_gap": max_gap,
        "max_marginal_identity_error": max_identity_error,
    }


def truncated_pmf(pmf: tuple[float, ...], k: int) -> np.ndarray:
    if k == 0:
        return np.array([1.0])
    out = np.zeros(k + 1)
    src = pad(pmf, max(len(pmf), k + 1))
    out[:k] = src[:k]
    out[k] = src[k:].sum()
    return out


def convolve_many(factors) -> np.ndarray:
    out = np.array([1.0])
    for factor in factors:
        out = np.convolve(out, factor)
    return out


def independent_population_dp(pmf: tuple[float, ...], budget: int, gamma: float):
    survival = np.r_[1.0, np.cumsum(np.asarray(pmf)[::-1])[::-1]]
    survival = np.array([1.0] + [sum(pmf[k:]) for k in range(1, budget + 2)])
    U = {(r, 0): 0.0 for r in range(budget + 1)}
    U.update({(0, n): 0.0 for n in range(budget + 1)})
    actions = {}
    for r in range(1, budget + 1):
        for n in range(1, budget + 1):
            values = []
            for s in range(r + 1):
                a, c = divmod(s, n)
                immediate = n * sum(survival[1 : a + 1]) + c * survival[a + 1]
                factors = [truncated_pmf(pmf, a)] * (n - c)
                factors += [truncated_pmf(pmf, a + 1)] * c
                trans = convolve_many(factors)
                continuation = sum(prob * U[(r - s, m)] for m, prob in enumerate(trans))
                values.append(immediate + gamma * continuation)
            actions[(r, n)] = int(np.argmax(values))
            U[(r, n)] = float(max(values))
    return U, actions


def dp_check(budgets, gamma: float):
    types = ((0.25, 0.25, 0.25, 0.25), (0.55, 0.05, 0.15, 0.25))
    mixture = tuple(np.mean(np.array(types), axis=0))
    rows = []
    for budget in budgets:
        population = CyclingPopulation(types)
        start = time.perf_counter()
        official = precompute_surrogate(budget, gamma, population, monte_carlo_samples=1200)
        official_seconds = time.perf_counter() - start
        start = time.perf_counter()
        oracle, actions = independent_population_dp(mixture, budget, gamma)
        oracle_seconds = time.perf_counter() - start
        errors = [abs(official.U[key] - value) for key, value in oracle.items()]
        rows.append({
            "budget": budget,
            "states_checked": len(oracle),
            "actions_checked": budget * budget * (budget + 3) // 2,
            "official_seconds": official_seconds,
            "oracle_seconds": oracle_seconds,
            "max_state_error": max(errors),
            "terminal_value": official.U[(budget, budget)],
        })
    return rows


def tv_pmf(p, q) -> float:
    length = max(len(p), len(q))
    return 0.5 * float(np.abs(pad(tuple(p), length) - pad(tuple(q), length)).sum())


def single_round_robustness():
    rng = np.random.default_rng(260512111)
    cases = 0
    max_violation = -np.inf
    for _ in range(1500):
        n = int(rng.integers(1, 6))
        budget = int(rng.integers(1, 9))
        true_pmfs = []
        estimated_pmfs = []
        for _ in range(n):
            p = tuple(rng.dirichlet(np.ones(budget + 1)))
            q = tuple(rng.dirichlet(np.ones(budget + 1)))
            true_pmfs.append(p)
            estimated_pmfs.append(q)
        true_alloc = official_greedy(true_pmfs, budget)
        noisy_alloc = official_greedy(estimated_pmfs, budget)
        loss = expected_reward(true_pmfs, true_alloc) - expected_reward(true_pmfs, noisy_alloc)
        bound = sum(
            abs(tails(p, budget) - tails(q, budget)).sum()
            for p, q in zip(true_pmfs, estimated_pmfs)
        )
        max_violation = max(max_violation, loss - bound)
        cases += 1

    tight = []
    for budget in (1, 2, 3, 5, 8, 13):
        x, beta = 0.8, 0.3
        # Bad person is index zero, so official np.argmax resolves the estimated tie adversarially.
        bad = tuple([1 - (x - beta)] + [0.0] * (budget - 1) + [x - beta])
        good = tuple([1 - x] + [0.0] * (budget - 1) + [x])
        tied = tuple([1 - (x - beta / 2)] + [0.0] * (budget - 1) + [x - beta / 2])
        true_pmfs = (bad, good)
        est_pmfs = (tied, tied)
        optimum = expected_reward(true_pmfs, official_greedy(true_pmfs, budget))
        selected = expected_reward(true_pmfs, official_greedy(est_pmfs, budget))
        bound = sum(abs(tails(p, budget) - tails(q, budget)).sum() for p, q in zip(true_pmfs, est_pmfs))
        tight.append({"budget": budget, "loss": optimum - selected, "bound": bound})
    return {
        "random_cases": cases,
        "max_bound_violation": max_violation,
        "tight_cases": tight,
        "max_tightness_error": max(abs(x["loss"] - x["bound"]) for x in tight),
    }


def outcome_distribution(pmfs, allocation):
    return convolve_many([truncated_pmf(p, k) for p, k in zip(pmfs, allocation)])


def exact_multiround_case(types, pop_probs, estimated_types, estimated_pop_probs, budget, gamma):
    type_count = len(types)
    mixture_hat = tuple(sum(estimated_pop_probs[t] * np.asarray(estimated_types[t]) for t in range(type_count)))
    U_hat, _ = independent_population_dp(mixture_hat, budget, gamma)

    @lru_cache(None)
    def frontier_probabilities(m):
        outcomes = []
        for ids in itertools.product(range(type_count), repeat=m):
            prob = float(np.prod([pop_probs[i] for i in ids]))
            outcomes.append((ids, prob))
        return outcomes

    @lru_cache(None)
    def optimal(r, ids):
        if r == 0 or len(ids) == 0:
            return 0.0
        pmfs = tuple(types[i] for i in ids)
        best = 0.0
        for s in range(1, r + 1):
            alloc = official_greedy(pmfs, s)
            trans = outcome_distribution(pmfs, alloc)
            immediate = expected_reward(pmfs, alloc)
            future = 0.0
            for m, pm in enumerate(trans):
                if pm:
                    future += pm * sum(prob * optimal(r - s, nxt) for nxt, prob in frontier_probabilities(m))
            best = max(best, immediate + gamma * future)
        return best

    @lru_cache(None)
    def policy(r, ids):
        if r == 0 or len(ids) == 0:
            return 0.0
        true_pmfs = tuple(types[i] for i in ids)
        hat_pmfs = tuple(estimated_types[i] for i in ids)
        scores = []
        hat_allocs = []
        for s in range(r + 1):
            alloc = official_greedy(hat_pmfs, s)
            hat_allocs.append(alloc)
            trans_hat = outcome_distribution(hat_pmfs, alloc)
            score = expected_reward(hat_pmfs, alloc) + gamma * sum(
                pm * U_hat[(r - s, m)] for m, pm in enumerate(trans_hat)
            )
            scores.append(score)
        s = int(np.argmax(scores))
        if s == 0:
            return 0.0
        alloc = hat_allocs[s]
        trans = outcome_distribution(true_pmfs, alloc)
        immediate = expected_reward(true_pmfs, alloc)
        future = 0.0
        for m, pm in enumerate(trans):
            if pm:
                future += pm * sum(prob * policy(r - s, nxt) for nxt, prob in frontier_probabilities(m))
        return immediate + gamma * future

    initial_ids = tuple(range(type_count))
    lhs = optimal(budget, initial_ids) - policy(budget, initial_ids)
    frontier_error = sum(tv_pmf(types[i], estimated_types[i]) for i in initial_ids)
    population_error = 0.5 * float(np.abs(np.asarray(pop_probs) - np.asarray(estimated_pop_probs)).sum())
    mean_pmf = sum(pop_probs[i] * np.asarray(types[i]) for i in range(type_count))
    heterogeneity = sum(pop_probs[i] * tv_pmf(types[i], mean_pmf) for i in range(type_count))
    c = 2 * gamma * budget / (1 - gamma)
    terms = {
        "frontier": 2 * (1 + gamma) * budget * frontier_error,
        "population": c * population_error,
        "heterogeneity": c * budget * heterogeneity,
    }
    return {"lhs_suboptimality": lhs, "rhs_bound": sum(terms.values()), "terms": terms}


def multi_round_robustness(budget: int, gamma: float):
    types = ((0.6, 0.3, 0.1), (0.15, 0.25, 0.6))
    exact = exact_multiround_case(types, (0.5, 0.5), types, (0.5, 0.5), budget, gamma)
    homogeneous = exact_multiround_case((types[0],), (1.0,), (types[0],), (1.0,), budget, gamma)
    noisy_types = (
        (0.927514479860685, 0.06615990750222918, 0.006325612637085679),
        (0.1802949832308228, 0.21324398160632396, 0.6064610351628531),
    )
    noisy = exact_multiround_case(
        types, (0.65, 0.35), noisy_types,
        (0.5006412138906983, 0.4993587861093018), budget, gamma
    )
    cases = {"heterogeneous_exact_models": exact, "homogeneous_zero_bound": homogeneous, "noisy_models": noisy}
    return {
        "cases": cases,
        "max_bound_violation": max(v["lhs_suboptimality"] - v["rhs_bound"] for v in cases.values()),
    }


def run(config_path: Path, output_dir: Path):
    cfg = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    claim1 = exhaustive_greedy_check(cfg["greedy_max_budget"])
    dp_rows = dp_check(cfg["dp_budgets"], cfg["gamma"])
    claim3_single = single_round_robustness()
    claim3_multi = multi_round_robustness(cfg["multi_round_budget"], cfg["gamma"])
    with (output_dir / "dp_scaling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dp_rows[0].keys())
        writer.writeheader()
        writer.writerows(dp_rows)
    summary = {
        "paper": "HigEPnWgLQ",
        "official_commit": "5e174a13e35cf03c57167c7c333193bd48745a93",
        "claim1_greedy_optimality": claim1,
        "claim2_population_dp": {
            "rows": dp_rows,
            "max_state_error": max(r["max_state_error"] for r in dp_rows),
            "largest_budget": max(cfg["dp_budgets"]),
        },
        "claim3_robustness": {"single_round": claim3_single, "multi_round": claim3_multi},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "repro/configs/full.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)

````


````json title=full.json
{
  "seed": 260512111,
  "greedy_max_budget": 8,
  "dp_budgets": [4, 6, 8, 10, 12, 16, 20],
  "gamma": 0.7,
  "multi_round_budget": 6
}


````


````output
{
  "paper": "HigEPnWgLQ",
  "official_commit": "5e174a13e35cf03c57167c7c333193bd48745a93",
  "claim1_greedy_optimality": {
    "frontier_budget_cases": 1881,
    "allocations_enumerated": 72609,
    "max_oracle_gap": 8.881784197001252e-16,
    "max_marginal_identity_error": 8.881784197001252e-16
  },
  "claim2_population_dp": {
    "rows": [
      {
        "budget": 4,
        "states_checked": 25,
        "actions_checked": 56,
        "official_seconds": 0.10167922102846205,
        "oracle_seconds": 0.001410794910043478,
        "max_state_error": 0.0,
        "terminal_value": 2.4
      },
      {
        "budget": 6,
        "states_checked": 49,
        "actions_checked": 162,
        "official_seconds": 0.12908055493608117,
        "oracle_seconds": 0.0034855150151997805,
        "max_state_error": 0.0,
        "terminal_value": 3.5999999999999996
      },
      {
        "budget": 8,
        "states_checked": 81,
        "actions_checked": 352,
        "official_seconds": 0.19489157292991877,
        "oracle_seconds": 0.008043833076953888,
        "max_state_error": 0.0,
        "terminal_value": 4.8
      },
      {
        "budget": 10,
        "states_checked": 121,
        "actions_checked": 650,
        "official_seconds": 0.30583862285129726,
        "oracle_seconds": 0.015681724064052105,
        "max_state_error": 0.0,
        "terminal_value": 6.0
      },
      {
        "budget": 12,
        "states_checked": 169,
        "actions_checked": 1080,
        "official_seconds": 0.47892355895601213,
        "oracle_seconds": 0.027170927030965686,
        "max_state_error": 0.0,
        "terminal_value": 7.199999999999999
      },
      {
        "budget": 16,
        "states_checked": 289,
        "actions_checked": 2432,
        "official_seconds": 1.061349306954071,
        "oracle_seconds": 0.0696402839384973,
        "max_state_error": 8.881784197001252e-16,
        "terminal_value": 9.6
      },
      {
        "budget": 20,
        "states_checked": 441,
        "actions_checked": 4600,
        "official_seconds": 2.0718385518994182,
        "oracle_seconds": 0.14189679501578212,
        "max_state_error": 1.7763568394002505e-15,
        "terminal_value": 12.0
      }
    ],
    "max_state_error": 1.7763568394002505e-15,
    "largest_budget": 20
  },
  "claim3_robustness": {
    "single_round": {
      "random_cases": 1500,
      "max_bound_violation": -0.004627134808053962,
      "tight_cases": [
        {
          "budget": 1,
          "loss": 0.30000000000000004,
          "bound": 0.30000000000000004
        },
        {
          "budget": 2,
          "loss": 0.6000000000000001,
          "bound": 0.6000000000000001
        },
        {
          "budget": 3,
          "loss": 0.9000000000000004,
          "bound": 0.9000000000000001
        },
        {
          "budget": 5,
          "loss": 1.5,
          "bound": 1.5000000000000002
        },
        {
          "budget": 8,
          "loss": 2.4000000000000004,
          "bound": 2.4000000000000004
        },
        {
          "budget": 13,
          "loss": 3.900000000000002,
          "bound": 3.8999999999999995
        }
      ],
      "max_tightness_error": 2.6645352591003757e-15
    },
    "multi_round": {
      "cases": {
        "heterogeneous_exact_models": {
          "lhs_suboptimality": 0.004329937500000103,
          "rhs_bound": 41.99999999999998,
          "terms": {
            "frontier": 0.0,
            "population": 0.0,
            "heterogeneity": 41.99999999999998
          }
        },
        "homogeneous_zero_bound": {
          "lhs_suboptimality": 0.0,
          "rhs_bound": 0.0,
          "terms": {
            "frontier": 0.0,
            "population": 0.0,
            "heterogeneity": 0.0
          }
        },
        "noisy_models": {
          "lhs_suboptimality": 0.38910442372275034,
          "rhs_bound": 49.8331641754494,
          "terms": {
            "frontier": 7.431118164388965,
            "population": 4.182046011060449,
            "heterogeneity": 38.219999999999985
          }
        }
      },
      "max_bound_violation": 0.0
    }
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_9513cacd4443", "created_at": "2026-07-17T04:36:18+00:00", "title": "Artifact: dp_scaling.csv", "path": "outputs/dp_scaling.csv", "size": 587, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/dp_scaling.csv` · dataset · 587 B

trackio-local-path://outputs/dp_scaling.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b442a409dc6c", "created_at": "2026-07-17T04:36:19+00:00", "title": "Verdict"}
-->
The single-round frontier-error inequality holds for **1,500** random distribution pairs. The paper’s adversarial tie family achieves equality for budgets `1,2,3,5,8,13` (maximum error `2.66e-15`). The full multi-round bound was independently checked by exact distribution-valued MDP enumeration: homogeneous models give `0 ≤ 0`; a heterogeneous case gives `0.00433 ≤ 42.0`; and a noisy case gives nontrivial loss `0.38910 ≤ 49.8332`, with frontier/population/heterogeneity terms recorded separately.
