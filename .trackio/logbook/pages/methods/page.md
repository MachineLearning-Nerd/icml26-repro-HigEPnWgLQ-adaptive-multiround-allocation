# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9825f667842c", "created_at": "2026-07-17T04:36:42+00:00", "title": "Independent verification design"}
-->
The reproduction executes the vendored official `OurPolicy.greedy_single_stage` and `precompute_surrogate` unchanged. Three independent mechanisms prevent circular validation:

1. every integer allocation is enumerated for Claim 1;
2. transition laws use direct `numpy.convolve`, never the official PGF helpers, for Claim 2;
3. a memoized exact MDP retains the complete tuple of frontier distribution types for Claim 3.

The finite population cycles deterministically through two types, making the official code’s 1,200-sample mixture exactly balanced rather than Monte Carlo noisy. All randomness is seeded.
