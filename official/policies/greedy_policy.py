# Standard library imports
from typing import cast

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from policies.abstract_policy import AbstractPolicy

class GreedyPolicy(AbstractPolicy):
    def __init__(
        self,
        rng_seed: int,
        gamma: float,
        population: AbstractPopulation,
        round_budget: int
    ) -> None:
        super().__init__(rng_seed, gamma, population)
        self.round_budget = round_budget
    
    def __repr__(self):
        return "GreedyPolicy"

    def generate_assignments(
        self,
        available_budget: int,
        distributions_or_realizations: list[AbstractArrivalDistribution | int]
    ) -> list[int]:
        budget = min(self.round_budget, available_budget)
        distributions = cast(list[AbstractArrivalDistribution], distributions_or_realizations)
        n = len(distributions)
        assignment = [0] * n
        for _ in range(budget):
            idx = np.argmax(
                [distributions[i].prob_at_least(assignment[i] + 1) for i in range(n)]
            )
            assignment[idx] += 1
        return assignment