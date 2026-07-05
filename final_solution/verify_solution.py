"""
Verification script for the Final Solution
============================================
Loads the (theta, M, X) found by final_solution.py and independently
checks the fit two ways:

1. Visual sanity check: overlay the fitted curve directly on top of the
   1500 scattered data points (plots/verify_overlay.png).
2. Quantitative check using the metric named in the assignment's
   assessment criteria: "L1 distance between uniformly sampled points
   between expected and predicted curve". Concretely:
     - Sample the fitted curve uniformly & densely over t in [6, 60].
     - For every real data point, find its nearest sample on that dense
       curve (Euclidean nearest-neighbor via KD-tree).
     - Report the L1 (|dx| + |dy|) distance between each data point and
       its nearest curve sample: mean, median, and max.
   This check is intentionally independent of the per-point t values
   recovered during the least-squares polish step in final_solution.py --
   it re-derives correspondence from scratch, so it acts as a genuine
   cross-check rather than re-reporting the same optimizer's own
   residuals.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

DATA_PATH = "../data/xy_data.csv"
PARAMS_PATH = "final_params.json"
T_MIN, T_MAX = 6, 60


def curve_xy(theta, M, X, t):
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - env * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + env * np.cos(theta)
    return x, y


def main():
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    theta, M, X = params["theta_rad"], params["M"], params["X"]

    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values

    # --- 1. Visual overlay ---
    t_plot = np.linspace(T_MIN, T_MAX, 3000)
    x_curve, y_curve = curve_xy(theta, M, X, t_plot)

    plt.figure(figsize=(8, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, label="data points (1500)")
    plt.plot(x_curve, y_curve, color="crimson", lw=1.5, label="fitted curve")
    plt.title(
        f"Fitted curve vs. data\ntheta={np.rad2deg(theta):.4f} deg, "
        f"M={M:.5f}, X={X:.4f}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/verify_overlay.png", dpi=130)
    print("Saved plots/verify_overlay.png")

    # --- 2. Independent L1 distance check ---
    t_dense = np.linspace(T_MIN, T_MAX, 20000)  # very dense, independent sampling
    x_dense, y_dense = curve_xy(theta, M, X, t_dense)
    curve_pts = np.column_stack([x_dense, y_dense])
    tree = cKDTree(curve_pts)
    dists, idx = tree.query(pts, k=1)

    nearest_curve_pts = curve_pts[idx]
    l1_per_point = np.abs(pts - nearest_curve_pts).sum(axis=1)

    print("\n--- L1 distance check (data points vs. nearest point on uniformly")
    print("    sampled fitted curve, assessment-criteria metric) ---")
    print(f"mean L1   = {l1_per_point.mean():.6f}")
    print(f"median L1 = {np.median(l1_per_point):.6f}")
    print(f"max L1    = {l1_per_point.max():.6f}")
    print(f"(Euclidean nearest-point distance -- mean={dists.mean():.6f}, "
          f"max={dists.max():.6f} -- for reference)")

    # --- 3. Residual histogram ---
    plt.figure(figsize=(6, 4))
    plt.hist(l1_per_point, bins=50, color="steelblue")
    plt.xlabel("L1 distance (data point -> nearest fitted-curve sample)")
    plt.ylabel("count")
    plt.title("Distribution of per-point L1 error (final solution)")
    plt.tight_layout()
    plt.savefig("plots/verify_l1_histogram.png", dpi=130)
    print("Saved plots/verify_l1_histogram.png")

    print("\n--- Final answer being verified ---")
    print(f"theta = {np.rad2deg(theta):.6f} deg  ({theta:.8f} rad)")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")


if __name__ == "__main__":
    main()
