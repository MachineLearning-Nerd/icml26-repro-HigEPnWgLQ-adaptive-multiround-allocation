# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_26a5f5932d85", "created_at": "2026-07-17T04:36:45+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T04:36:46+00:00"}
-->
**Outcome: all three scored claims are verified.** Official greedy search is exhaustive-oracle optimal over 72,609 allocations; the official PGF dynamic program agrees with independent direct convolution to `1.78e-15`; and both the tight single-round error term and full multi-round decomposition survive exact adversarial checks. No result depends on GPU hardware or an inaccessible benchmark.

## Scope & cost

| | Scope | Hardware | Time | Cost | Outcome |
|---|---|---|---|---|---|
| This reproduction | 1,881 complete greedy cases; all DP states through b=20; exact MDP through b=6 | 4-vCPU host, CPU only | ~12 s evidence + tests | $0 | 3/3 verified |
| Full empirical replication | ICPSR networks, 9 SLURM array jobs × 30 trials | authors request 64 CPUs/job, 64 GB, up to 24 h | cluster-scale | not incurred | Separate experiment, not a scored claim |

Artifacts: `outputs/summary.json`, `outputs/dp_scaling.csv`, executed commands, official pinned source, and 19 passing tests.
