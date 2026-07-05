# Approach 3 — Brute-force grid search

## Why I chose this approach

Approach 2 (PCA closed-form estimate) got close but not exact, and it
relies on an assumption (uniform spacing of points in `t`) that
introduces bias. Before building a proper optimizer, I wanted to check
the simplest possible alternative: since the allowed ranges for
`theta`, `M`, and `X` are small and fully known, why not just evaluate a
dense grid of candidates directly and keep the best one? A grid search
can't get trapped the way a single local optimizer can (there's no
"starting point" to get stuck near) — it looks everywhere by
construction, so it's a natural thing to try before something more
sophisticated.

The scoring function is the same nearest-point (Chamfer-style) distance
used in Approach 2's evaluation and in the Final Solution: sample the
candidate curve densely for `t` in `[6, 60]`, build a KD-tree over the
samples, and for every real data point measure the distance to its
nearest curve sample. Mean squared distance is the score to minimize.

## What I did

Ran the exact same grid search at increasing resolution `n` (`n` values
per axis, so `n^3` total combinations), timing each run:

| n (per axis) | total combos | time measured | best theta | best M | best X | loss |
|---|---|---|---|---|---|---|
| 10 | 1,000 | 3.3 s | 33.33° | 0.0389 | 55.56 | 1.83 |
| 20 | 8,000 | 25.7 s | 28.95° | 0.0289 | 52.63 | 1.25 |
| 40 | 64,000 | 205.9 s (~3.4 min) | 29.49° | 0.0295 | 53.85 | 0.30 |

(`n=40` is not included in the committed script because it takes several
minutes; `n=10` and `n=20` are kept so the script runs quickly, and the
`n=40` numbers above are from an actual measured run.)

## Result

The trend is clear and exactly what the "curse of dimensionality"
predicts: **the score keeps improving as the grid gets finer, but it
never actually reaches zero, and the cost grows as `n³`.**

Extrapolating from the measured `n=40` timing (0.00322 s/combination):
to get anywhere near the precision the final answer needs (matching
`theta`, `M`, `X` to within numerical noise, not just "close enough to
look right"), the grid would realistically need on the order of
`n ≈ 1000` steps per axis — i.e. **1,000,000,000 combinations**, which
at the measured per-combination cost would take:

```
1e9 combos * 0.00322 s/combo  ≈  3.2 * 10^6 s  ≈  37 days
```

on a single core, just to get one digit of extra precision on each
parameter. That's before even accounting for the fact that a *fixed*
grid can never land exactly on the true values anyway — the answer
almost certainly falls between grid points, so there's always a residual
"quantization" error no matter how long you wait.

## Inference / why I moved on

Grid search confirmed something useful: **the loss landscape does
improve smoothly as the grid refines** — there's no evidence here of the
badly-behaved, many-false-minima landscape that a small periodic term
like `sin(0.3t)` could in principle create (see the note in the Final
Solution's README about verifying this more directly with multiple
optimizer starts). But brute force is simply the wrong tool for
*precision* in continuous parameter spaces — it scales terribly with the
number of unknowns and the precision demanded.

What's needed is a search strategy that:
1. Still looks broadly across the whole parameter space (grid search's
   one good property), but
2. Converges continuously to a precise, non-quantized answer instead of
   being capped by a fixed step size.

That combination — global exploration + continuous convergence — is
exactly what a population-based global optimizer like **Differential
Evolution** provides, which is why it's the core of the Final Solution.
