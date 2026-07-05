# Roadmap: From the Problem Statement to the Final Answer

This is the full story of how this assignment was solved — the
mathematical understanding first, then every approach tried in the order
it was tried, including the ones that didn't work and exactly what each
one taught me before moving to the next.

---

## 0. The problem, restated

We're given a parametric curve:

```
x(t) = t*cos(theta) - e^(M|t|)*sin(0.3t)*sin(theta) + X
y(t) = 42 + t*sin(theta) + e^(M|t|)*sin(0.3t)*cos(theta)
```

with three unknowns `theta`, `M`, `X` inside known ranges
(`0°<theta<50°`, `-0.05<M<0.05`, `0<X<100`), a known parameter range for
`t` (`6<t<60`), and a CSV of 1500 `(x, y)` points that lie somewhere on
this curve for `t` in that range. The task is to recover the exact values
of `theta`, `M`, `X`.

## 1. Mathematical insight, from scratch

**What is a parametric curve?** Instead of `y = f(x)` directly, both
coordinates are written as functions of a third variable `t` (imagine `t`
as time, and `(x(t), y(t))` as the position of a dot moving along the
curve).

**What do `theta`, `M`, `X` actually do, geometrically?**

Group the formula by separating the `t` term from the wiggle term:

```
x - X   =  t*cos(theta)  -  E*sin(theta)
y - 42  =  t*sin(theta)  +  E*cos(theta)          where E = e^(M|t|)*sin(0.3t)
```

- `(cos(theta), sin(theta))` and `(-sin(theta), cos(theta))` are
  **orthonormal vectors** — this is exactly a 2D rotation matrix. So
  `theta` spins the whole curve around the origin by that angle, and in
  the *rotated* frame, one axis carries `t` directly and the perpendicular
  axis carries `E`.
- `E = e^(M|t|)*sin(0.3t)` is an oscillation (`sin(0.3t)`, period ≈ 20.9
  in `t`) with a changing amplitude (`e^(M|t|)`, an *envelope*): if
  `M > 0` the wiggles grow as `t` grows; if `M < 0` they shrink.
- `X` is a plain horizontal shift of the whole curve (`42` is given, not
  unknown — a fixed vertical shift).

So geometrically: find the rotation, wiggle-decay rate, and horizontal
offset that make the formula trace exactly through the 1500 given points.

**Loss function / optimization, briefly.** A loss function is a single
number scoring "how wrong" a guess is — low is good. Optimizers try to
make this number smaller. A **local** optimizer starts somewhere and
walks downhill until it can't improve, like a ball rolling into the
nearest dip; if the loss landscape has many small dips and only one true
lowest point, a local optimizer can get stuck in the wrong one. The
`sin(0.3t)` term is exactly the kind of thing that *can* make a loss
landscape bumpy like this (small rotations can shift the wiggly parts of
the curve in and out of alignment with the data), which turns out to be
the central risk to design around here (see Approach 3 → Final Solution,
and the bonus multi-start cross-check, which found this failure mode
*actually happening* for one starting point out of ten).

## 2. Progression of approaches

Full detail, code, and results for each of these live in their own
folder under `initial_approaches/`, each with its own README explaining
why it was tried, what happened, and what it fed into the next step.

### Approach 1 — "The CSV rows are ordered by t" (`initial_approaches/approach_1_ordered_assumption/`)

**Idea:** row 0 → `t≈6`, row 1499 → `t≈60`, evenly spaced; plug straight
into `scipy.optimize.curve_fit`.
**Result:** the sanity-check plot of `x`/`y` against assumed `t` is
jagged and noisy — not the smooth curve it should be if the assumption
held — and the forced fit has an RMS residual of ~16 units (huge, on
data spanning only ~50-110 in `x`).
**Insight:** the rows are shuffled. Every later approach has to treat the
1500 points as an **unordered cloud** — no relying on row position as a
stand-in for `t`.

### Approach 2 — Closed-form estimation via PCA (`initial_approaches/approach_2_pca_statistical_estimation/`)

**Idea:** exploit the orthogonal rotation structure derived above. Since
covariance is translation-invariant, Principal Component Analysis on the
raw, unordered `(x, y)` cloud recovers `(cos(theta), sin(theta))` as its
top principal direction directly — no ordering, no `X`, no iterative
search needed. Project onto that axis to approximate each point's `t`,
project onto the perpendicular axis to approximate `E`, then a tiny 1-D
`curve_fit` solves for `M`, and averaging recovers `X`.
**Result:** `theta≈28.48°`, `M≈0.0288`, `X≈54.58` — genuinely close (all
three land in the right neighborhood) but not exact; mean squared
nearest-point distance ≈ 0.46, max gap ≈ 1.3 units.
**Insight:** the closed-form estimate is biased by an assumption (roughly
uniform sampling of `t`) and by the radial/perpendicular axes not being
perfectly independent. It's an excellent, cheap **initial guess** but not
precise enough on its own — the natural next question is how to refine
it exactly.

### Approach 3 — Brute-force grid search (`initial_approaches/approach_3_bruteforce_grid_search/`)

**Idea:** the allowed ranges are small and fully known, so evaluate a
dense grid of `(theta, M, X)` combinations directly (using the same
KD-tree-based nearest-point loss as Approach 2's evaluation) and keep the
best.
**Result:** measured at increasing resolution — `n=10` (1,000 combos,
3.3s, loss 1.83), `n=20` (8,000 combos, 25.7s, loss 1.25), `n=40`
(64,000 combos, 206s, loss 0.30). The score keeps improving but never
reaches zero, and cost grows as `n³`. Extrapolating the measured
per-combination cost to a grid fine enough for real precision
(`n≈1000` per axis, ~1 billion combinations) would take roughly **37
days** on one core.
**Insight:** grid search *does* search broadly (good — no local-minimum
risk), but it scales terribly with precision demanded ("curse of
dimensionality"). What's needed is something that searches broadly like a
grid but converges continuously and precisely like a local optimizer.

### Final Solution — Differential Evolution + Chamfer Loss + Joint Least-Squares Polish (`final_solution/`)

**Idea:** combine the good half of every earlier attempt:
- the correspondence-free **nearest-point (Chamfer) loss**, evaluated
  fast via a **KD-tree** (kept from Approaches 2 & 3's evaluation),
- **broad exploration** of the whole parameter space (kept from Approach
  3's spirit, but via a population-based global optimizer —
  **Differential Evolution** — instead of a fixed grid, so it converges
  continuously instead of being capped by a grid step size),
- a final **joint least-squares polish** that solves for `theta`, `M`,
  `X`, *and* an individual `t_i` for every one of the 1500 points
  simultaneously (1503 unknowns at once), starting from the nearest
  curve-sample match found via the KD-tree. This directly asks "is there
  an exact `t` for every point that makes this curve fit perfectly?"

**Result:**
```
Differential Evolution: theta=29.99961 deg, M=0.030000, X=54.99961, loss=0.0000217
Joint LS polish:         theta=29.999973 deg, M=0.030000, X=54.999998
                         max|residual|=9.56e-06, RMS residual=1.91e-06
```
Independent verification (fresh 20,000-point curve resampling,
`final_solution/verify_solution.py`) gives a mean L1 distance of
`0.000999` between data points and the fitted curve, and the overlay plot
shows the fitted curve running exactly through the scattered data with no
visible gap. A bonus multi-start cross-check
(`final_solution/extra_bonus_analysis/multi_start_cross_check.py`) found
9 of 10 wildly different starting points converge to this same answer via
an independent local optimizer — and the 10th genuinely got trapped in a
local minimum, which is real, observed confirmation of exactly the risk
that motivated using a global optimizer in the first place. A bonus
uncertainty analysis
(`final_solution/extra_bonus_analysis/parameter_uncertainty.py`) puts
95% confidence intervals around each parameter at roughly `1e-7`,
consistent with the data having been generated from exact values with no
added noise.

## 3. Final Answer

```
theta = 30 deg  =  0.5235987755982988 rad   (= pi/6)
M     = 0.03
X     = 55
```

Desmos-ready parametric expression (domain `6 <= t <= 60`):

```
\left(t*\cos(0.5235987755982988)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987755982988)+55,42+t*\sin(0.5235987755982988)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987755982988)\right)
```

## 4. Repository structure

```
.
├── roadmap.md                                 <- this file
├── requirements.txt
├── data/
│   └── xy_data.csv                            <- the given dataset
├── initial_approaches/
│   ├── approach_1_ordered_assumption/
│   │   ├── approach1_ordered_assumption.py
│   │   └── README.md
│   ├── approach_2_pca_statistical_estimation/
│   │   ├── approach2_pca_estimation.py
│   │   └── README.md
│   └── approach_3_bruteforce_grid_search/
│       ├── approach3_grid_search.py
│       └── README.md
└── final_solution/
    ├── final_solution.py                      <- DE + Chamfer loss + LS polish -> final_params.json
    ├── verify_solution.py                     <- independent L1 check + overlay plot
    ├── final_params.json                      <- generated by final_solution.py
    ├── plots/
    │   ├── verify_overlay.png
    │   └── verify_l1_histogram.png
    ├── extra_bonus_analysis/
    │   ├── multi_start_cross_check.py         <- bonus: confirms global optimum
    │   └── parameter_uncertainty.py           <- bonus: Jacobian-based confidence intervals
    └── README.md
```

## 5. How to reproduce

```bash
pip install -r requirements.txt

# Initial approaches (run from inside each folder, paths are relative)
cd initial_approaches/approach_1_ordered_assumption && python3 approach1_ordered_assumption.py
cd ../approach_2_pca_statistical_estimation && python3 approach2_pca_estimation.py
cd ../approach_3_bruteforce_grid_search && python3 approach3_grid_search.py

# Final solution
cd ../../final_solution
python3 final_solution.py        # writes final_params.json, ~30s
python3 verify_solution.py       # writes plots/, ~5s

# Bonus analysis
cd extra_bonus_analysis
python3 multi_start_cross_check.py
python3 parameter_uncertainty.py
```
