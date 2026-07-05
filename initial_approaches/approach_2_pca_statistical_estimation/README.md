# Approach 2 — Closed-form estimation via PCA (no optimizer, no ordering needed)

## Why I chose this approach

Approach 1 established that the 1500 points are an **unordered cloud** —
I know each point lies somewhere on the curve, but not which `t` produced
it. Before reaching for a heavy iterative optimizer, I wanted to see how
much could be recovered directly from the *geometry* of the formula
itself, because a good closed-form estimate is extremely valuable later
as a warm-start / cross-check for whatever optimizer I use next.

Rewriting the given formula by grouping the `t` term and the wiggle term
separately:

```
x - X   =  t * cos(theta)   -   E * sin(theta)
y - 42  =  t * sin(theta)   +   E * cos(theta)         where E = e^(M|t|) * sin(0.3t)
```

`(cos θ, sin θ)` and `(−sin θ, cos θ)` are **orthonormal** vectors — this
is exactly a rotated coordinate system. In that rotated frame, one axis
carries `t` (which ranges over a wide interval, 6 to 60) and the
perpendicular axis carries `E` (a bounded oscillation, since
`|sin(0.3t)| ≤ 1` and `M` is small). Because `t` has far more spread than
`E`, the point cloud should be visibly stretched along the
`(cos θ, sin θ)` direction — which is exactly what Principal Component
Analysis finds: the direction of maximum variance. Crucially, covariance
is translation-invariant, so this works **without knowing `X` or even
using the given `42` offset.**

## What I did

1. Computed the covariance matrix of the raw `(x, y)` point cloud and
   took its top eigenvector as an estimate of `(cos θ, sin θ)` → gives
   `theta` directly.
2. Projected every point onto that principal axis, sorted the
   projections, and linearly mapped the sorted values onto the known
   range `[6, 60]` — this recovers an approximate `t` for every point
   *without ever using row order*.
3. Projected onto the perpendicular axis to get an approximate `E` per
   point, then ran a tiny 1-D `curve_fit` of
   `E ≈ exp(M|t|) * sin(0.3t)` to solve for the single remaining unknown,
   `M`.
4. Backed out `X` by averaging `x_i − t_i·cos θ + E_i·sin θ` over all
   points.

## Result

```
theta = 28.4831 deg   (true value turned out to be 30 deg)
M     = 0.02884       (true value turned out to be 0.03)
X     = 54.5763        (true value turned out to be 55)

Mean squared nearest-point distance = 0.461
Max nearest-point distance          = 1.313
```

This is genuinely close — well inside the allowed parameter ranges and
in the right neighborhood on all three unknowns — but it is **not
exact**. A max nearest-point gap of ~1.3 units is easily visible if you
overlay the fitted curve on the data.

## Inference / why I still needed another step

Two approximations are baked into this method and both introduce small,
systematic errors:

1. **Uniform-t assumption.** Mapping sorted projections linearly onto
   `[6, 60]` assumes the 1500 points are evenly spread across `t`. If the
   true sampling isn't perfectly uniform (or if points landed unevenly
   because of the oscillation), the recovered `t_i` values are slightly
   off, which biases the `M` fit.
2. **Imperfect axis separation.** The "radial" and "perpendicular"
   projections aren't perfectly independent — because `E` is itself a
   function of `t`, there is a small correlation between the two axes
   that PCA doesn't account for, nsudging `theta` away from its true
   value.

So this approach is excellent as a **fast, correspondence-free initial
guess** — it gets all three unknowns into roughly the right basin with
zero iterative search — but it isn't precise enough to submit as a final
answer. The natural next step is to treat `(theta ≈ 28.5°, M ≈ 0.029,
X ≈ 54.6)` as a starting point for a proper optimizer, and separately,
to make sure that optimizer is *global* rather than a single local
descent (see Approach 3 for why a naive search strategy is also
insufficient, and the Final Solution for the method that actually
closes this remaining gap to numerical zero).
