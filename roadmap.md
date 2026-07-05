# Roadmap: Problem Statement to Final Answer

A structured account of the mathematical analysis and iterative approach progression undertaken to solve this assignment. Each approach is documented in order of execution, including unsuccessful attempts and the specific insights each contributed to the next.

---

## 0. Problem Statement

Given a parametric curve:

```
x(t) = t·cos(theta) − e^(M|t|)·sin(0.3t)·sin(theta) + X
y(t) = 42 + t·sin(theta) + e^(M|t|)·sin(0.3t)·cos(theta)
```

with three unknowns `theta`, `M`, `X` constrained to known ranges (`0° < theta < 50°`, `−0.05 < M < 0.05`, `0 < X < 100`), a known parameter domain `6 < t < 60`, and a CSV of 1500 `(x, y)` points sampled from this curve. The objective is to recover the exact values of `theta`, `M`, and `X`.

## 1. Mathematical Analysis

**Parametric curve structure.** Both coordinates are expressed as functions of a third variable `t`. Grouping the linear and oscillatory components:

```
x − X   =  t·cos(theta)  −  E·sin(theta)
y − 42  =  t·sin(theta)  +  E·cos(theta)          where E = e^(M|t|)·sin(0.3t)
```

- `(cos(theta), sin(theta))` and `(−sin(theta), cos(theta))` form **orthonormal vectors** — a 2D rotation matrix. The parameter `theta` rotates the entire curve; in the rotated frame, one axis carries `t` directly and the perpendicular axis carries `E`.
- `E = e^(M|t|)·sin(0.3t)` is an oscillation (`sin(0.3t)`, period ≈ 20.9 in `t`) with a varying amplitude envelope (`e^(M|t|)`): positive `M` causes growing oscillations; negative `M` causes decaying ones.
- `X` is a horizontal translation of the entire curve; `42` is a given, fixed vertical offset.

Geometrically: the task is to determine the rotation angle, envelope decay rate, and horizontal offset that cause the parametric formula to trace exactly through the 1500 given data points.

**Optimization landscape considerations.** The `sin(0.3t)` term introduces the risk of a bumpy loss landscape — small parameter perturbations can shift oscillatory features in and out of alignment with the data, creating local minima. This risk is confirmed empirically by the multi-start cross-check in the Final Solution, where 1 of 10 starting points became trapped in a local minimum.

## 2. Approach Progression

Full detail, code, and results for each approach reside in dedicated subdirectories under `initial_approaches/`, each with its own README.

### Approach 1 — Ordered Assumption (`initial_approaches/approach_1_ordered_assumption/`)

**Hypothesis:** Row 0 → `t ≈ 6`, row 1499 → `t ≈ 60`, evenly spaced; fit via `scipy.optimize.curve_fit`.
**Result:** Sanity-check plot of `x`/`y` against assumed `t` is jagged and noisy. Forced fit yields RMS residual of ~16 units (data spans only ~50–110 in `x`).
**Insight:** The rows are shuffled. All subsequent approaches must treat the data as an **unordered point cloud** with no row-position-to-`t` correspondence.

### Approach 2 — PCA-Based Closed-Form Estimation (`initial_approaches/approach_2_pca_statistical_estimation/`)

**Method:** Exploit the orthogonal rotation structure via Principal Component Analysis on the unordered `(x, y)` cloud to recover `(cos(theta), sin(theta))` as the top principal direction — no ordering or knowledge of `X` required. Project onto axes to approximate `t` and `E` per point; 1-D `curve_fit` for `M`; averaging for `X`.
**Result:** `theta ≈ 28.48°`, `M ≈ 0.0288`, `X ≈ 54.58` — all in the correct neighbourhood but not exact. Mean squared nearest-point distance ≈ 0.46, max gap ≈ 1.3 units.
**Insight:** Biased by the uniform-`t` assumption and imperfect axis separation. Serves as an excellent **initial estimate** but lacks sufficient precision for a final answer.

### Approach 3 — Brute-Force Grid Search (`initial_approaches/approach_3_bruteforce_grid_search/`)

**Method:** Evaluate a dense grid of `(theta, M, X)` combinations using KD-tree-based nearest-point loss.
**Result:** At increasing resolution — `n=10` (1,000 combos, 3.3 s, loss 1.83), `n=20` (8,000 combos, 25.7 s, loss 1.25), `n=40` (64,000 combos, 206 s, loss 0.30). Loss improves monotonically but never reaches zero; cost scales as O(n³). Extrapolation to sufficient precision (~10⁹ combinations) yields an estimated runtime of **~37 days** on a single core.
**Insight:** Grid search provides broad exploration without local-minimum risk, but scales prohibitively with required precision. The needed combination is global exploration with continuous convergence.

### Final Solution — Differential Evolution + Chamfer Loss + Joint Least-Squares Polish (`final_solution/`)

**Method:** Combines the strengths of all prior approaches:
- Correspondence-free **nearest-point (Chamfer) loss** via KD-tree (from Approaches 2 & 3)
- **Broad exploration** via population-based global optimizer — **Differential Evolution** — instead of a fixed grid (from Approach 3's intent, without its scaling limitation)
- **Joint least-squares polish** solving for `theta`, `M`, `X`, *and* an individual `t_i` for each of the 1500 points simultaneously (1503 unknowns), collapsing residuals to numerical noise

**Result:**
```
Differential Evolution: theta=29.99961°, M=0.030000, X=54.99961, loss=0.0000217
Joint LS Polish:        theta=29.999973°, M=0.030000, X=54.999998
                        max|residual|=9.56e-06, RMS residual=1.91e-06
```
Independent verification (20,000-point curve resampling) yields mean L1 distance of 0.000999. Multi-start cross-check confirms convergence from 9 of 10 starting points; the 10th becomes trapped in a local minimum (loss ≈ 449), empirically validating the necessity of a global optimizer. Uncertainty analysis places 95% confidence intervals at ~10⁻⁷, consistent with noise-free data generation.

## 3. Final Answer

```
theta = 30 deg  =  0.5235987755982988 rad   (= π/6)
M     = 0.03
X     = 55
```

Desmos-ready parametric expression (domain `6 ≤ t ≤ 60`):

```
\left(t*\cos(0.5235987755982988)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987755982988)+55,42+t*\sin(0.5235987755982988)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987755982988)\right)
```

## 4. Repository Structure

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

## 5. Reproduction Instructions

```bash
pip install -r requirements.txt

# Initial approaches (run from inside each folder; paths are relative)
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
