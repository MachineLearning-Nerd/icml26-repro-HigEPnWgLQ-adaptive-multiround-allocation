# Reproduction: Adaptive Multi-Round Allocation with Stochastic Arrivals

This CPU reproduction verifies all three scored claims for ICML 2026 paper `HigEPnWgLQ` using the official implementation pinned at `5e174a1` and independent exhaustive oracles.

## Results

1. **Exact single-round greedy allocation — verified.** The unchanged official greedy method matches exhaustive optimization in all 1,881 frontier/budget cases, covering 72,609 integer allocations. Maximum objective gap is `8.88e-16`.
2. **Exact polynomial population DP — verified.** The official truncated-PGF table matches an independently implemented direct-convolution Bellman recursion across seven budgets through `b=20`, including 441 states at the largest budget. Maximum state error is `1.78e-15`; the table has quadratic state count and the executed action/state loops are polynomial in `b`.
3. **Robustness decomposition and tight frontier term — verified.** The single-round inequality holds in 1,500 random noisy-frontier cases. The paper's adversarial tie construction attains the bound at six budgets through 13 (maximum equality error `2.66e-15`). An independent exact multi-round MDP verifies the full three-term bound for homogeneous, heterogeneous, and noisy population models; the noisy case has actual suboptimality `0.3891`, below its bound.

## Run

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python repro/src/run_allocation.py --config repro/configs/full.json --output-dir outputs
pytest -q
```

Raw results are in `outputs/summary.json` and `outputs/dp_scaling.csv`.

## Scope

The paper's separate ICPSR network experiment is not one of the three scored claims and is intentionally excluded because its source data require an external download. This reproduction targets the exact claims directly: all allocations in a broad deterministic suite, every population-DP state through budget 20, and exact multi-round value functions through budget 6. No GPU is used.

