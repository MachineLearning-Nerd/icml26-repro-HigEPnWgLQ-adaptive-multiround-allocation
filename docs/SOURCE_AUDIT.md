# Source and scope audit

- Paper: *Adaptive Multi-Round Allocation with Stochastic Arrivals*, arXiv:2605.12111, OpenReview `HigEPnWgLQ`.
- Official repository: `cxjdavin/Adaptive-Multi-Round-Allocation-with-Stochastic-Arrivals`.
- Vendored commit: `5e174a13e35cf03c57167c7c333193bd48745a93`.
- Relevant official paths: `policies/our_policy.py`, `core/population_distribution_object.py`, and `core/utils.py`.

The scored claims are exact algorithmic/theoretical statements, so the reproduction exercises complete finite state spaces rather than the paper's separate ICPSR experiment. The official greedy and population-DP implementations are executed unchanged. Independent oracles enumerate every allocation, convolve transition distributions without the official PGF utilities, and solve the small true multi-round MDP over distribution-valued frontiers.

The official population object estimates a mixture by Monte Carlo. Our finite population cycles through two distribution types, and the requested 1,200 samples are exactly balanced. Thus the mixture passed into the unchanged official DP is exact, not noisy.

