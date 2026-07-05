"""
Approach 3: Brute-force grid search
====================================
Idea:
    Approach 2 gave a good but imprecise closed-form estimate. The
    allowed ranges for the unknowns are small and fully known:
        0 deg < theta < 50 deg
        -0.05 < M < 0.05
        0 < X < 100
    So instead of trusting a single starting guess for a local optimizer,
    why not just evaluate a dense grid of (theta, M, X) combinations and
    keep whichever scores lowest against the data? This guarantees the
    whole space is looked at, unlike a single-point local search.

Loss function used (same nearest-point / Chamfer-style distance used in
Approach 2's evaluation and in the Final Solution): for a candidate
(theta, M, X), generate a dense set of points along the candidate curve
for t in [6, 60], build a KD-tree over them, and for every real data
point find the distance to its nearest generated curve point. The mean
of the squared distances is the score to minimize.

What this script demonstrates:
    Running the SAME grid search at increasing resolution (n points per
    axis) to make the core trade-off concrete: a coarse grid is fast but
    imprecise, and making it fine enough to be precise makes it
    computationally infeasible (the "curse of dimensionality" -- cost
    grows as n^3 for 3 unknowns).
"""

import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree

DATA_PATH = "../../data/xy_data.csv"
T_MIN, T_MAX = 6, 60
N_CURVE_SAMPLES = 2000


def chamfer_loss(theta, M, X, pts, t_dense):
    env = np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    xr = t_dense * np.cos(theta) - env * np.sin(theta) + X
    yr = 42 + t_dense * np.sin(theta) + env * np.cos(theta)
    curve_pts = np.column_stack([xr, yr])
    tree = cKDTree(curve_pts)
    d, _ = tree.query(pts, k=1)
    return np.mean(d ** 2)


def run_grid(n, pts, t_dense):
    thetas = np.linspace(0, np.deg2rad(50), n)
    Ms = np.linspace(-0.05, 0.05, n)
    Xs = np.linspace(0, 100, n)

    best_val = np.inf
    best_params = None
    t0 = time.time()
    for th in thetas:
        for m in Ms:
            for x in Xs:
                val = chamfer_loss(th, m, x, pts, t_dense)
                if val < best_val:
                    best_val = val
                    best_params = (th, m, x)
    dt = time.time() - t0
    return best_params, best_val, dt, n ** 3


def main():
    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values
    t_dense = np.linspace(T_MIN, T_MAX, N_CURVE_SAMPLES)

    # NOTE: n=40 (64,000 combinations) takes ~3-4 minutes on a single core.
    # Kept small here (10, 20) so the script finishes quickly; n=40's
    # measured result is reported in the README for the full picture.
    for n in [10, 20]:
        (theta, M, X), val, dt, n_combos = run_grid(n, pts, t_dense)
        print(
            f"grid n={n:>3} ({n_combos:>7} combos): "
            f"time={dt:6.2f}s  theta={np.rad2deg(theta):6.2f} deg  "
            f"M={M:7.4f}  X={X:6.2f}  loss={val:.4f}"
        )

    print("\nSee README.md for the n=40 measurement and the extrapolation")
    print("to a grid fine enough to reach the precision the final answer needs.")


if __name__ == "__main__":
    main()
