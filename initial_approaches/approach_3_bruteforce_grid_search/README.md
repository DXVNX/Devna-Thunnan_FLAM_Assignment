# Approach 3 — Brute-Force Grid Search

## Rationale

Approach 2 (PCA estimation) produced a close but biased estimate. Before constructing a sophisticated optimizer, it is worth evaluating the simplest exhaustive alternative: since the allowed ranges for `theta`, `M`, and `X` are small and fully specified, a dense grid of candidate combinations can be evaluated directly, retaining the best. Grid search has no starting-point dependency and therefore cannot get trapped in local minima — it searches the entire space by construction.

The scoring function is the same nearest-point (Chamfer-style) distance used in Approach 2's evaluation: sample the candidate curve densely for `t ∈ [6, 60]`, build a KD-tree, and for each data point measure the Euclidean distance to its nearest curve sample. Mean squared distance serves as the loss to minimize.

## Methodology

The grid search was executed at increasing resolution `n` (values per axis, `n³` total combinations), with timing recorded for each run:

| n (per axis) | Total Combos | Time | Best θ | Best M | Best X | Loss |
|---|---|---|---|---|---|---|
| 10 | 1,000 | 3.3 s | 33.33° | 0.0389 | 55.56 | 1.83 |
| 20 | 8,000 | 25.7 s | 28.95° | 0.0289 | 52.63 | 1.25 |
| 40 | 64,000 | 205.9 s (~3.4 min) | 29.49° | 0.0295 | 53.85 | 0.30 |

(`n=40` is excluded from the committed script due to runtime; `n=10` and `n=20` are retained for quick reproducibility. The `n=40` figures above are from an actual measured run.)

## Results

The trend follows the expected "curse of dimensionality" pattern: **loss improves monotonically with grid resolution but never reaches zero, while computational cost scales as O(n³).**

Extrapolating from the measured `n=40` timing (0.00322 s/combination): achieving precision sufficient to match the final answer (parameters resolved to within numerical noise) would require approximately `n ≈ 1000` steps per axis — i.e., **10⁹ combinations**:

```
10⁹ combos × 0.00322 s/combo  ≈  3.2 × 10⁶ s  ≈  37 days (single core)
```

Furthermore, a fixed grid can never land exactly on the true continuous parameter values; there is always a residual quantization error regardless of runtime.

## Key Takeaway

Grid search confirms that the **loss landscape improves smoothly with resolution** — no evidence of the pathological multi-modal structure that the periodic `sin(0.3t)` term could theoretically induce. However, brute force is fundamentally unsuitable for precision in continuous parameter spaces due to cubic scaling.

The requirements for the next approach are:
1. **Broad exploration** of the full parameter space (grid search's strength), and
2. **Continuous convergence** to a precise, non-quantized answer (grid search's weakness).

This combination — global exploration with continuous refinement — is exactly what a population-based global optimizer such as **Differential Evolution** provides, forming the core of the Final Solution.
