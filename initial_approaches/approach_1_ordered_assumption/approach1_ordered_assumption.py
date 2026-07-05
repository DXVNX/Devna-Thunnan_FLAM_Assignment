"""
Approach 1: Naive "rows are ordered by t" assumption
======================================================
Idea:
    The CSV has 1500 rows and the problem states points are given "for
    6 < t < 60". The simplest possible assumption is that row 0 -> t ~ 6,
    row 1499 -> t ~ 60, evenly spaced in between. If true, this collapses
    to a textbook curve_fit problem: build t = linspace(6, 60, 1500),
    plug (t, x, y) into scipy.optimize.curve_fit, and solve directly.

Why it seemed reasonable:
    It is the simplest possible interpretation of "here is a list of
    points on the curve", and many toy datasets *are* given in generation
    order.

Result of this script:
    - A sanity-check plot of x and y against the assumed t (row index).
    - A curve_fit attempt using that assumed t.

See README.md in this folder for the outcome and why this approach was
abandoned.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_PATH = "../../data/xy_data.csv"


def model_xy(t, theta, M, X):
    """The given parametric model, returned as separate x(t), y(t)."""
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - env * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + env * np.cos(theta)
    return x, y


def stacked_model(t, theta, M, X):
    """curve_fit needs a single flat output vector, so x and y are
    stacked into one array: [x_1..x_n, y_1..y_n]."""
    x, y = model_xy(t, theta, M, X)
    return np.concatenate([x, y])


def main():
    df = pd.read_csv(DATA_PATH)
    x_data = df["x"].values
    y_data = df["y"].values
    n = len(x_data)

    # --- Step 1: assume row order corresponds to t order ---
    t_assumed = np.linspace(6, 60, n)

    # --- Step 2: sanity check. If the assumption is right, x(t) and y(t)
    # should look like smooth, gently-curving lines when plotted against
    # the assumed t / row index. ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_assumed, x_data, ".", ms=2)
    axes[0].set_title("x vs assumed t (row order)")
    axes[0].set_xlabel("assumed t")
    axes[0].set_ylabel("x")
    axes[1].plot(t_assumed, y_data, ".", ms=2)
    axes[1].set_title("y vs assumed t (row order)")
    axes[1].set_xlabel("assumed t")
    axes[1].set_ylabel("y")
    plt.tight_layout()
    plt.savefig("ordering_sanity_check.png", dpi=110)
    print("Saved ordering_sanity_check.png -- inspect this first.")

    # --- Step 3: fit anyway, to quantify just how bad the assumption is ---
    y_stacked = np.concatenate([x_data, y_data])
    theta0 = np.deg2rad(25)  # midpoint of allowed range, as a neutral start
    popt, _ = curve_fit(
        stacked_model,
        t_assumed,
        y_stacked,
        p0=[theta0, 0.0, 50.0],
        bounds=([0, -0.05, 0], [np.deg2rad(50), 0.05, 100]),
        maxfev=20000,
    )
    theta_fit, M_fit, X_fit = popt
    resid = stacked_model(t_assumed, *popt) - y_stacked
    rms = np.sqrt(np.mean(resid ** 2))

    print("\n--- Fit result under the 'ordered rows' assumption ---")
    print(f"theta = {np.rad2deg(theta_fit):.4f} deg")
    print(f"M     = {M_fit:.5f}")
    print(f"X     = {X_fit:.4f}")
    print(f"RMS residual = {rms:.4f}  (data scale is ~10-100, so this is huge)")


if __name__ == "__main__":
    main()
