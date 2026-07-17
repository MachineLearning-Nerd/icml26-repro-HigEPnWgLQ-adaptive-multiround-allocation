# Standard library imports
import pickle

from pathlib import Path
from typing import Any, Union

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution

def construct_polynomial_from_individual_distribution(
        distribution: AbstractArrivalDistribution,
        max_degree: int
    ) -> np.ndarray:
        coeffs = [distribution.prob_equal(j) for j in range(max_degree)]
        coeffs.append(distribution.prob_at_least(max_degree))
        return np.array(coeffs)

def multiply_PGFs_up_to_s(pgf1: np.ndarray, pgf2: np.ndarray, s: int) -> np.ndarray:
    return truncate_poly(np.polynomial.polynomial.polymul(pgf1, pgf2), s)

def power_PGFs_up_to_s(pgf: np.ndarray, pow: int, s: int) -> np.ndarray:
    result = np.array([1.0])
    x = truncate_poly(pgf, s)
    e = pow
    while e > 0:
        if e & 1:
            result = multiply_PGFs_up_to_s(result, x, s)
        e >>= 1
        if e:
            x = multiply_PGFs_up_to_s(x, x, s)
    return truncate_poly(result, s)

def truncate_poly(poly: np.ndarray, s: int) -> np.ndarray:
    out = np.zeros(s+1, dtype=float)
    k = min(s+1, len(poly))
    out[:k] = poly[:k]
    if len(poly) > s:
        out[s] = poly[s:].sum()
    assert np.isclose(1, out.sum())
    return out

"""
Save an object as a Pickle file. Creates directories if needed.
"""
def save_pickle(obj: Any, path: Union[str, Path]) -> None: 
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

"""
Load an object from a Pickle file.
"""
def load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
