"""
Approach 2: Closed-form statistical estimation via PCA
=======================================================
Idea:
    Approach 1 showed the rows carry no ordering information, so t cannot
    be assumed. But we don't have to throw away structure entirely -- the
    *shape* of the formula tells us something useful:

        x - X   =  t*cos(theta)          - E*sin(theta)
        y - 42  =  t*sin(theta)          + E*cos(theta)

    where E = exp(M*|t|) * sin(0.3*t). Notice that (cos(theta), sin(theta))
    and (-sin(theta), cos(theta)) are ORTHONORMAL vectors -- this is
    literally a rotated coordinate frame. In that frame:
        - the "radial" axis carries the value t directly
        - the "perpendicular" axis carries the oscillation E

    Since t ranges over a wide interval (6 to 60) while E is a bounded
    oscillation (|sin(0.3t)| <= 1, and M is small), the point cloud should
    be visibly elongated along the (cos theta, sin theta) direction. That
    means Principal Component Analysis (PCA) on the raw (x, y) cloud
    should recover theta directly, with NO knowledge of point order,
    t-values, or even X.

    Once theta is known, we can:
      1. Project every point onto the two PCA axes to get an approximate
         "radial" coordinate (~ t) and "perpendicular" coordinate (~ E).
      2. Rank-order the radial coordinates and map them linearly onto the
         known range [6, 60] (this recovers each point's approximate t,
         entirely without needing the CSV order).
      3. Fit the single unknown M with a 1-D curve_fit of the
         perpendicular coordinate against exp(M|t|) * sin(0.3t).
      4. Back out X by averaging x_i - t_i*cos(theta) + E_i*sin(theta)
         over all points.

Why it's useful even though it's approximate:
    It needs no iterative search at all and gives a very good initial
    guess "for free", straight from linear algebra + one 1-D fit. It's
    also useful as a sanity check / cross-validation for whatever the
    final optimizer converges to.

Result of this script:
    Printed (theta, M, X) estimates plus a Chamfer-distance style error
    score against the data, to quantify how close the closed-form
    estimate gets before any nonlinear optimization is used.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

DATA_PATH = "../../data/xy_data.csv"
T_MIN, T_MAX = 6, 60


def estimate_theta_via_pca(pts):
    """Top principal component of the raw point cloud approximates the
    direction (cos(theta), sin(theta)), because covariance is invariant
    to the (unknown) translation (X, 42)."""
    mean_xy = pts.mean(axis=0)
    centered = pts - mean_xy
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    v1 = eigvecs[:, -1]  # top principal direction
    v2 = eigvecs[:, -2]  # orthogonal complement
    # theta is known to be in (0, 50) deg -> cos>0 and sin>0, use that to
    # fix the sign ambiguity inherent to eigenvectors.
    if v1[0] < 0:
        v1 = -v1
        v2 = -v2
    theta_est = np.arctan2(v1[1], v1[0])
    return theta_est, v1, v2, mean_xy


def estimate_t_via_projection(pts, mean_xy, v1, n_points):
    """Project onto the principal axis, then linearly map the ranks onto
    the known t range [T_MIN, T_MAX]. This is only valid if the 1500
    points are roughly uniformly distributed across the t range, which is
    the natural assumption for a generated dataset ("points that lie on
    the curve for 6 < t < 60")."""
    centered = pts - mean_xy
    r = centered @ v1
    order = np.argsort(r)
    t_grid = np.linspace(T_MIN, T_MAX, n_points)
    t_est = np.empty(n_points)
    t_est[order] = t_grid
    return t_est, (centered @ np.array([-v1[1], v1[0]]))  # also return perp proj for reuse if needed


def main():
    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values
    n = len(pts)

    # --- Step 1: theta from PCA ---
    theta_est, v1, v2, mean_xy = estimate_theta_via_pca(pts)
    print(f"theta_est (PCA)      = {np.rad2deg(theta_est):.4f} deg")

    # --- Step 2: approximate t per point from the radial projection ---
    centered = pts - mean_xy
    r = centered @ v1
    s = centered @ v2
    order = np.argsort(r)
    t_est = np.empty(n)
    t_est[order] = np.linspace(T_MIN, T_MAX, n)

    # --- Step 3: fit M from the perpendicular ("wiggle") component ---
    def env_model(t, M, s_offset):
        return np.exp(M * np.abs(t)) * np.sin(0.3 * t) - s_offset

    popt, _ = curve_fit(env_model, t_est, s, p0=[0.0, 0.0], maxfev=10000)
    M_est, s_offset = popt
    print(f"M_est                = {M_est:.5f}")

    # --- Step 4: back out X by averaging over all points ---
    E_est = np.exp(M_est * np.abs(t_est)) * np.sin(0.3 * t_est)
    X_candidates = pts[:, 0] - t_est * np.cos(theta_est) + E_est * np.sin(theta_est)
    X_est = X_candidates.mean()
    print(f"X_est                = {X_est:.4f}  (std across points: {X_candidates.std():.4f})")

    # --- Quantify how good this closed-form estimate already is ---
    t_dense = np.linspace(T_MIN, T_MAX, 4000)
    env = np.exp(M_est * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    xr = t_dense * np.cos(theta_est) - env * np.sin(theta_est) + X_est
    yr = 42 + t_dense * np.sin(theta_est) + env * np.cos(theta_est)
    curve_pts = np.column_stack([xr, yr])
    tree = cKDTree(curve_pts)
    dists, _ = tree.query(pts, k=1)

    print("\n--- Closed-form (PCA + 1-D fit) estimate ---")
    print(f"theta = {np.rad2deg(theta_est):.4f} deg")
    print(f"M     = {M_est:.5f}")
    print(f"X     = {X_est:.4f}")
    print(f"Mean squared nearest-point distance = {np.mean(dists**2):.5f}")
    print(f"Max nearest-point distance           = {dists.max():.5f}")


if __name__ == "__main__":
    main()
