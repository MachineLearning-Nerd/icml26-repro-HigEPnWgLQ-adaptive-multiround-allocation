# Standard library imports
import os
import sys
import time

from copy import deepcopy
from multiprocessing import Pool

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# Local imports
from core.ICPSR22140_processor import ICPSR22140Processor
from core.empirical_arrival_distribution import EmpiricalArrivalDistribution
from core.empirical_population import EmpiricalPopulation
from core.utils import load_pickle, save_pickle
from policies.abstract_policy import AbstractPolicy
from policies.constant_policy import ConstantPolicy
from policies.greedy_policy import GreedyPolicy
from policies.greedy_remainder_policy import GreedyRemainderPolicy
from policies.our_policy import OurPolicy, precompute_surrogate

def _fit_tree_empirical_predictor(X, y, min_fraction) -> tuple[np.ndarray, dict]:
    n = X.shape[0]
    min_leaf = max(1, int(np.floor(min_fraction * n)))
    clf = DecisionTreeClassifier(
        min_samples_leaf=min_leaf,
        max_depth=None,
    )
    clf.fit(X, y)

    # Compute leaf ID for each sample and store node_type and type_to_node dicts
    leaf_ids = np.asarray(clf.apply(X), dtype=int)

    # Build empirical distributions for each leaf
    empirical_distributions = dict()
    for leaf in np.unique(leaf_ids):
        y_leaf = y[leaf_ids == leaf]
        n_leaf = len(y_leaf)
        empirical_distributions[leaf] = {int(c): (y_leaf == c).sum() / n_leaf for c in clf.classes_}

    return leaf_ids, empirical_distributions

def preprocess_ICPSR(std_name: str, threshold: float) -> tuple[nx.Graph, dict, dict, dict, dict]:
    tsv_file1 = "data/22140-0001-Data.tsv"
    tsv_file2 = "data/22140-0002-Data.tsv"
    tsv_file3 = "data/22140-0003-Data.tsv"
    pickle_filename = "ICPSR_22140.pkl"
    processor = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename)
    covariates, statuses, G, _, _, _ = processor.merged_datasets[std_name]

    node_ids = []
    X_train = []
    y_train = []
    for idx, cov in covariates.items():
        node_ids.append(idx)
        X_train.append(cov + [statuses[idx]])
        y_train.append(G.degree[idx])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    leaf_ids, empirical_distributions = _fit_tree_empirical_predictor(X_train, y_train, min_fraction=threshold)

    node_type = {node_ids[i]: int(leaf_ids[i]) for i in range(len(node_ids))}
    type_to_nodes = dict()
    for i, leaf in enumerate(leaf_ids):
        leaf = int(leaf)
        node = node_ids[i]
        type_to_nodes.setdefault(leaf, []).append(node)
    
    denom = sum(len(lst) for lst in type_to_nodes.values())
    type_proportion = {key: len(type_to_nodes[key])/denom for key in type_to_nodes.keys()}

    assert all(u in node_type for u in G.nodes())
    return G, node_type, type_to_nodes, type_proportion, empirical_distributions

def _build_policy(
    policy_str: str,
    rng_seed: int,
    gamma: float,
    population,
    policy_param,
    max_budget: int,
) -> AbstractPolicy:
    if policy_str == "ConstantPolicy":
        return ConstantPolicy(rng_seed, gamma, population, policy_param)
    if policy_str == "GreedyPolicy":
        return GreedyPolicy(rng_seed, gamma, population, int(policy_param * max_budget))
    if policy_str == "GreedyRemainderPolicy":
        return GreedyRemainderPolicy(rng_seed, gamma, population, policy_param)
    if policy_str == "OurPolicy":
        t0 = time.perf_counter()
        surrogate_object = precompute_surrogate(max_budget, gamma, population)
        surrogate_seconds = time.perf_counter() - t0
        print(
            f"precompute_surrogate (real job): {surrogate_seconds:.4f}s "
            f"(max_budget={max_budget}, gamma={gamma})",
            flush=True,
        )
        return OurPolicy(rng_seed, gamma, population, surrogate_object)
    raise NotImplementedError(f"Unknown policy: {policy_name}")

def _simulate_synthetic_job(
    policy,
    population,
    initial_frontier_size: int,
    max_budget: int,
    verbose: bool,
) -> list[int]:
    system_size_over_time = [initial_frontier_size]
    arriving_size = initial_frontier_size
    available_budget = max_budget
    timestep = 1

    while arriving_size > 0 and available_budget > 0:
        distributions = population.sample_arrival_distributions(arriving_size)
        if verbose:
            print(
                f"\nTimestep {timestep}: {len(distributions)} distributions, "
                f"{available_budget} remaining budget"
            )

        # Compute assignments using policy
        assignments = policy.generate_assignments(available_budget, deepcopy(distributions))

        # Simulate realizations
        realizations = [dist.sample() for dist in distributions]

        # Update budget
        assert sum(assignments) <= available_budget
        available_budget -= sum(assignments)

        # Next batch of arrivals
        arriving_size = sum(
            min(realizations[i], assignments[i]) for i in range(len(distributions))
        )

        if verbose:
            print(
                f"{sum(assignments)} budget assigned to {len(assignments)} people "
                f"in round {timestep}: {assignments}"
            )
            print(f"Realizations: {realizations}")
            print(f"Reward of round {timestep}: {arriving_size}")

        system_size_over_time.append(system_size_over_time[-1] + arriving_size)
        timestep += 1

    if verbose:
        if available_budget == 0:
            print("\nOut of budget!")
        if arriving_size == 0:
            print("\nNo more arrivals!")

    return system_size_over_time

def _simulate_graph_job(
    policy: AbstractPolicy,
    rng_seed: int,
    G: nx.Graph,
    node_type: dict,
    empirical_distributions: dict,
    initial_frontier_size: int,
    max_budget: int,
    verbose: bool,
) -> list[int]:
    rng = np.random.default_rng(rng_seed)
    system_size_over_time = [initial_frontier_size]
    frontier = [
        int(x)
        for x in rng.choice(list(G.nodes()), initial_frontier_size, replace=False)
    ]
    in_system = set(frontier)
    initial_frontier = set(frontier)
    available_budget = max_budget
    timestep = 1

    while len(frontier) > 0 and available_budget > 0:
        # Define distributions based on types
        rng_seeds = [int(rng.integers(int(1e6))) for _ in frontier]
        distributions = []
        for i, idx in enumerate(frontier):
            if idx in initial_frontier:
                empirical_distribution = empirical_distributions[node_type[idx]]
            else:
                # Reduce degree by 1 since at least one neighbor is already in the system
                empirical_distribution = {
                    (k - 1): v
                    for k, v in empirical_distributions[node_type[idx]].items()
                    if isinstance(k, int) and k >= 1
                }

            distributions.append(
                EmpiricalArrivalDistribution(rng_seeds[i], empirical_distribution)
            )
        if verbose:
            print(
                f"\nTimestep {timestep}: {len(distributions)} distributions, "
                f"{available_budget} remaining budget"
            )

        # Compute assignments using policy
        assignments = policy.generate_assignments(available_budget, deepcopy(distributions))

        # Update budget
        assert sum(assignments) <= available_budget
        available_budget -= sum(assignments)

        # Next batch of arrivals (recruit neighbors)
        candidates = [set(G.neighbors(idx)) - in_system for idx in frontier]
        available = set().union(*candidates)
        new_frontier = []
        for i in rng.permutation(len(frontier)):
            if assignments[i] <= 0:
                continue
            feasible = list(candidates[i] & available)
            k = min(assignments[i], len(feasible))
            if k > 0:
                recruited = rng.choice(feasible, size=k, replace=False).tolist()
                new_frontier.extend(recruited)
                available.difference_update(recruited)

        frontier = list(set(new_frontier))
        in_system.update(frontier)

        if verbose:
            print(
                f"{sum(assignments)} budget assigned to {len(assignments)} people "
                f"in round {timestep}: {assignments}"
            )
            print(f"Realizations: {len(frontier)}")
            print(f"Reward of round {timestep}: {len(frontier)}")

        system_size_over_time.append(system_size_over_time[-1] + len(frontier))
        timestep += 1

    if verbose:
        if available_budget == 0:
            print("\nOut of budget!")
        if len(frontier) == 0:
            print("\nNo more arrivals!")

    return system_size_over_time

def run_job(job_parameters) -> tuple[tuple[str, int], np.ndarray]:
    fname = job_parameters['fname']
    sim = job_parameters['sim']
    gamma = job_parameters['gamma']
    max_budget = job_parameters['max_budget']
    population = job_parameters['population']
    initial_frontier_size = job_parameters['initial_frontier_size']
    policy_str = job_parameters['policy']
    policy_param = job_parameters['policy_param']
    rng_seed = job_parameters['rng_seed']
    verbose = job_parameters['verbose']
    G = job_parameters['G']
    node_type = job_parameters['node_type']
    empirical_distributions = job_parameters['empirical_distributions']

    if not os.path.isfile(fname):
        policy = _build_policy(
            policy_str,
            rng_seed=rng_seed,
            gamma=gamma,
            population=population,
            policy_param=policy_param,
            max_budget=max_budget,
        )

        if sim:
            system_size_over_time = _simulate_synthetic_job(
                policy=policy,
                population=population,
                initial_frontier_size=initial_frontier_size,
                max_budget=max_budget,
                verbose=verbose,
            )
        else:
            system_size_over_time = _simulate_graph_job(
                policy=policy,
                rng_seed=rng_seed,
                G=G,
                node_type=node_type,
                empirical_distributions=empirical_distributions,
                initial_frontier_size=initial_frontier_size,
                max_budget=max_budget,
                verbose=verbose,
            )

        # Save results to pickle
        res = (str(policy), policy_param), np.array(system_size_over_time)
        save_pickle(res, fname)

    # Load results from pickle
    res = load_pickle(fname)

    return res

def run_experiment(
    num_times: int,
    max_budget: int,
    gamma: float,
    initial_frontier_size: int,
    rng_seed: int = 42,
    multithread: bool = True
) -> None:
    stds = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    all_policy_pairs = [
        ("OurPolicy", None),
        ("ConstantPolicy", 2),
        ("ConstantPolicy", 3),
        ("ConstantPolicy", 5),
        ("ConstantPolicy", 10),
        ("GreedyPolicy", 0.1),
        ("GreedyPolicy", 0.2),
        ("GreedyPolicy", 0.3),
        ("GreedyPolicy", 0.5),
        ("GreedyPolicy", 1.0),
        ("GreedyRemainderPolicy", 0.1),
        ("GreedyRemainderPolicy", 0.2),
        ("GreedyRemainderPolicy", 0.3),
        ("GreedyRemainderPolicy", 0.5),
        ("GreedyRemainderPolicy", 1.0),
    ]
    # One aggregate file per (num_times, budget, gamma, n, seed) so Slurm array tasks do not clobber each other.
    result_fname = (
        f"results/full_nt{num_times}_b{max_budget}_g{gamma}_n{initial_frontier_size}_seed{rng_seed}.pkl"
    )

    if os.path.isfile(result_fname):
        print(f"Skip: {result_fname} already exists", flush=True)
        return

    # {(std_name, sim): {(str(policy), policy_param): [trajectory arrays]}}
    all_results: dict[tuple[str, bool], dict[tuple[str, int | None], list]] = {}

    for std in stds:
        for sim in [True, False]:
            G, node_type, _, type_proportion, empirical_distributions = preprocess_ICPSR(std, 0.01)
            exp_type = "ICPSR_sim" if sim else "ICPSR_real"
            param_string = f"{std}_{sim}_{num_times}_{max_budget}_{gamma}_{initial_frontier_size}_{rng_seed}"

            trajectories = {(str(policy), policy_param): [] for policy, policy_param in all_policy_pairs}
            jobs = []
            for policy, policy_param in all_policy_pairs:
                rng = np.random.default_rng(rng_seed)
                populations = [
                    EmpiricalPopulation(
                        int(rng.integers(int(1e6))), type_proportion, empirical_distributions
                    )
                    for _ in range(num_times)
                ]
                for t in range(num_times):
                    job_parameters = dict()
                    job_parameters["fname"] = (
                        f"results/{policy}_{policy_param}_{param_string}_{t}.pkl"
                    )
                    job_parameters["sim"] = sim
                    job_parameters["max_budget"] = max_budget
                    job_parameters["gamma"] = gamma
                    job_parameters["initial_frontier_size"] = initial_frontier_size
                    job_parameters["population"] = populations[t]
                    job_parameters["policy"] = policy
                    job_parameters["policy_param"] = policy_param
                    job_parameters["rng_seed"] = int(rng.integers(int(1e6)))
                    job_parameters["verbose"] = False
                    job_parameters["G"] = G
                    job_parameters["node_type"] = node_type
                    job_parameters["empirical_distributions"] = empirical_distributions
                    jobs.append(job_parameters)

            if multithread:
                with Pool() as pool:
                    for policy_key, system_size_over_time in tqdm(
                        pool.imap_unordered(run_job, jobs),
                        total=len(jobs),
                        desc=f"{exp_type} {std}",
                    ):
                        trajectories[policy_key].append(system_size_over_time)
            else:
                for job in tqdm(jobs, desc=f"{exp_type} {std}"):
                    policy_key, system_size_over_time = run_job(job)
                    trajectories[policy_key].append(system_size_over_time)

            all_results[(std, sim)] = trajectories

    save_pickle(all_results, result_fname)

if __name__ == "__main__":
    num_times = int(sys.argv[1])
    max_budget = int(sys.argv[2])
    gamma = float(sys.argv[3])
    initial_frontier_size = int(sys.argv[4])
    
    print(f"{num_times} runs, b = {max_budget}, gamma = {gamma}, n = {initial_frontier_size}")
    run_experiment(num_times, max_budget, gamma, initial_frontier_size)
