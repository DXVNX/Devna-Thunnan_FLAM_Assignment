# Approach 2 — Closed-Form Estimation via PCA

## Rationale

Approach 1 established that the 1500 data points form an **unordered cloud** with no usable row ordering. Before resorting to iterative optimization, the geometric structure of the parametric formula itself can be exploited to obtain a closed-form estimate — valuable both as a standalone approximation and as a warm-start for subsequent refinement.

Rewriting the parametric equations by separating the linear and oscillatory components:

```
x - X   =  t * cos(theta)   -   E * sin(theta)
y - 42  =  t * sin(theta)   +   E * cos(theta)         where E = e^(M|t|) * sin(0.3t)
```

The vectors `(cos θ, sin θ)` and `(−sin θ, cos θ)` are **orthonormal** — this is a 2D rotation. In the rotated frame, one axis carries `t` (ranging over [6, 60]) and the perpendicular axis carries `E` (a bounded oscillation). Since `t` has far greater spread than `E`, the point cloud is elongated along the `(cos θ, sin θ)` direction — precisely what Principal Component Analysis recovers as its top eigenvector. Crucially, covariance is translation-invariant, so this works **without knowing `X`**.

## Methodology

1. Computed the covariance matrix of the raw `(x, y)` cloud; extracted the top eigenvector as an estimate of `(cos θ, sin θ)` → `theta` directly.
2. Projected each point onto this principal axis, sorted the projections, and linearly mapped the sorted values onto `[6, 60]` — recovering an approximate `t` per point without using row order.
3. Projected onto the perpendicular axis to obtain approximate `E` values, then ran a 1-D `curve_fit` of `E ≈ exp(M|t|) * sin(0.3t)` to solve for `M`.
4. Recovered `X` by averaging `x_i − t_i·cos θ + E_i·sin θ` across all points.

## Results

```
theta = 28.4831 deg   (true: 30 deg)
M     = 0.02884       (true: 0.03)
X     = 54.5763       (true: 55)

Mean squared nearest-point distance = 0.461
Max nearest-point distance          = 1.313
```

All three parameters land in the correct neighborhood — well within the allowed ranges — but the fit is **not exact**. A maximum nearest-point gap of ~1.3 units is clearly visible in an overlay plot.

## Key Takeaway

Two systematic biases are inherent to this method:

1. **Uniform-t assumption.** Mapping sorted projections linearly onto `[6, 60]` assumes even distribution of points across `t`. Non-uniform sampling or clustering due to the oscillatory term introduces bias in the recovered `t_i` values, which propagates into the `M` estimate.
2. **Imperfect axis separation.** Since `E` is itself a function of `t`, a small correlation exists between the two projected axes that PCA does not account for, nudging `theta` away from its true value.

This approach serves as an excellent **fast, correspondence-free initial estimate** — all three unknowns are placed in roughly the correct basin with zero iterative search. However, it lacks the precision required for a final answer. The natural next step is to use these estimates as a starting region for a proper optimizer, preferably one with global search capability (see Approach 3 for why naive exhaustive search is also insufficient, and the Final Solution for the method that closes this gap to numerical zero).
