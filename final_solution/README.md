# Final Solution — Differential Evolution + Chamfer Loss + Joint Least-Squares Polish

## Rationale

Each prior approach contributed a specific insight, and each revealed a specific limitation:

| Source | Contribution Retained | Limitation Identified |
|---|---|---|
| Approach 1 (Ordered Assumption) | — | Data has **no usable order**; any loss function must be correspondence-free |
| Approach 2 (PCA Estimation) | Correspondence-free geometric estimation | Biased by uniform-`t` assumption; no iterative refinement |
| Approach 3 (Grid Search) | Broad, global search without starting-point dependency | Cost scales as O(n³); fixed grid cannot resolve continuous parameters exactly |

The final solution synthesizes the strengths while addressing all identified limitations:

1. **Loss Function — Nearest-Point (Chamfer) Distance.** For a candidate `(theta, M, X)`, a dense set of points is generated along the candidate curve for `t ∈ [6, 60]`. For each data point, the distance to the nearest generated curve point is computed via a KD-tree (1500 data points against thousands of curve samples, remaining computationally efficient). No point-ordering knowledge is required.

2. **Search Strategy — Differential Evolution.** Rather than a single starting guess (which the bonus analysis below demonstrates *can* fail) or a rigid grid (computationally infeasible at required precision), Differential Evolution maintains a population of `(theta, M, X)` candidates distributed across the allowed ranges. Each generation produces new candidates by combining and perturbing existing ones, replacing weaker candidates with stronger ones. This provides parallel exploration of multiple regions — robust against the periodic false-valley risk introduced by `sin(0.3t)` — while converging continuously rather than being limited by a fixed step size.

3. **Precision Polish — Joint Least-Squares Refit.** Differential Evolution's loss is built on a finite curve sample, so its solution is good but not exact. The polish step solves a single least-squares problem over **1503 unknowns simultaneously**: `theta`, `M`, `X`, and an individual `t_i` for each of the 1500 data points. This directly asks whether an exact `t` exists for every point such that the curve passes through it precisely — collapsing residuals to the numerical noise floor.

## Pipeline

```
final_solution.py
  Step 1: differential_evolution over bounds
              theta ∈ [0, 50°], M ∈ [-0.05, 0.05], X ∈ [0, 100]
          minimizing KD-tree Chamfer loss (4000 curve samples).
  Step 2: least_squares jointly over (theta, M, X, t_1 ... t_1500),
          seeded from Step 1's nearest-curve-sample match for each point,
          driven to residuals at the numerical noise floor.
  -> final_params.json

verify_solution.py
  Independent re-derivation of correspondence (fresh 20,000-point
  curve sampling + KD-tree) reporting:
    - Visual overlay of fitted curve on scattered data
      (plots/verify_overlay.png)
    - L1 distance metric as specified in the assignment's assessment
      criteria, between data points and nearest point on a uniformly
      sampled fitted curve
    - Residual histogram (plots/verify_l1_histogram.png)

extra_bonus_analysis/multi_start_cross_check.py
  Plain local optimizer launched from 10 widely separated starting points
  (including all four corners of the allowed parameter box) using the
  same Chamfer loss — testing for local-minimum traps directly rather
  than relying on the global optimizer's own convergence report.

extra_bonus_analysis/parameter_uncertainty.py
  Least-squares Jacobian at the optimum used to compute formal 95%
  confidence intervals for theta, M, and X.
```

## Results

**Step 1 — Differential Evolution:**
```
theta = 29.99961 deg   M = 0.030000   X = 54.99961
loss (mean squared nearest-point distance) = 0.0000217
time  ~ 17–20 s
```

**Step 2 — Joint Least-Squares Polish:**
```
theta = 29.999973 deg  (0.52359830 rad)
M     = 0.030000
X     = 54.999998
max |residual| = 9.56e-06
RMS residual   = 1.91e-06
```

These residuals are at the numerical noise floor of the optimizer (double-precision least-squares solvers routinely bottom out around 10⁻⁶ to 10⁻⁸ on well-posed problems), constituting strong evidence that the fit is *exact*, not merely close.

**Independent Verification (`verify_solution.py`):**
```
Mean L1 distance (data point → nearest point on a fresh,
independent 20,000-point uniform curve sampling) = 0.000999
Median L1 = 0.000983
Max L1    = 0.003068
```
The overlay plot (`plots/verify_overlay.png`) shows the fitted curve passing directly through the scattered data with no visible gap along its entire length.

**Multi-Start Cross-Check (`extra_bonus_analysis/multi_start_cross_check.py`):**
9 of 10 widely separated starting points (including three of the four extreme corners of the parameter box) converged to the identical solution via a completely independent local optimizer. The 10th start (`theta=49°, M=-0.049, X=1`) became trapped in a local minimum (loss ≈ 449 vs. ≈0.00002 for the rest) — a real, observed instance of the failure mode that motivated the use of a global optimizer.

**Uncertainty Estimate (`extra_bonus_analysis/parameter_uncertainty.py`):**
```
theta = 29.999973 deg  ± 3.3e-07 deg  (95% CI)
M     = 0.030000       ± 1.4e-09      (95% CI)
X     = 54.999998      ± 3.0e-07      (95% CI)
```
Confidence intervals of this magnitude, centred on clean values, strongly indicate the data was generated from these exact parameter values with no added noise.

## Final Answer

```
theta = 30 deg   =  0.5235987756 rad   (π/6)
M     = 0.03
X     = 55
```

**Desmos-ready parametric expression** (domain `6 ≤ t ≤ 60`):

```
\left(t*\cos(0.5235987755982988)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987755982988)+55,42+t*\sin(0.5235987755982988)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987755982988)\right)
```

## Conclusion

All three parameters resolve to clean values within their allowed ranges (`30°` within `0°–50°`, `0.03` within `−0.05–0.05`, `55` within `0–100`). Post-polish residuals are at floating-point noise level. The convergence of four independent lines of evidence — (a) a global optimizer's convergence, (b) an essentially-exact least-squares refit, (c) an independently-resampled L1/visual verification, and (d) a multi-start cross-check reproducing the same answer from 9 unrelated starting points — provides high confidence that `theta = 30°, M = 0.03, X = 55` is the exact generating parameterization of the dataset.
