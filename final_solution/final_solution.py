"""
Final Solution: Differential Evolution + Chamfer Loss + Joint Least-Squares Polish
====================================================================================
This combines the useful pieces from every earlier attempt while fixing
their specific flaws:

  - From Approach 2 (PCA):        a correspondence-free loss function
                                   (nearest-point / Chamfer distance) that
                                   never needs to know point order.
  - From Approach 3 (grid search): the idea of searching BROADLY across
                                   the whole parameter space instead of
                                   trusting one starting guess.
  - New in this step:              a population-based GLOBAL optimizer
                                   (Differential Evolution) that searches
                                   broadly like a grid but converges
                                   continuously and precisely like a local
                                   optimizer -- getting the good half of
                                   both earlier approaches without their
                                   respective weaknesses (imprecision /
                                   combinatorial blow-up).
  - A final "polish" step:         a joint least-squares refit that solves
                                   for theta, M, X AND an individual t_i
                                   for every one of the 1500 points at
                                   once. This directly asks "is there an
                                   exact t for every point that makes this
                                   curve fit perfectly?", squeezing out the
                                   last bit of error the KD-tree-based
                                   Chamfer loss can't fully resolve (since
                                   it only ever checks a *finite* sample of
                                   the curve, not the true continuous
                                   curve).

Pipeline
--------
1. Load the 1500 unordered (x, y) points.
2. Define the Chamfer loss: for a candidate (theta, M, X), sample the
   curve densely over t in [6, 60], build a KD-tree over the samples, and
   score = mean squared distance from each real point to its nearest
   curve sample.
3. Run scipy.optimize.differential_evolution over the full allowed
   ranges to find a global minimum of that loss.
4. Use the DE result's nearest-curve-sample match to seed an initial `t`
   guess for every point, then run scipy.optimize.least_squares jointly
   over (theta, M, X, t_1, ..., t_1500) to polish the fit to numerical
   precision.
5. Report final (theta, M, X), residual statistics, and save the values
   for the verification script.
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial import cKDTree

DATA_PATH = "../data/xy_data.csv"
T_MIN, T_MAX = 6, 60
N_CURVE_SAMPLES_DE = 4000  # dense enough that DE's loss landscape is a
                            # faithful, smooth stand-in for the true curve


# Module-level globals so this loss function is picklable and can be used
# with differential_evolution's multiprocess workers (workers=-1). A
# closure would not survive pickling across worker processes.
_PTS = None
_T_DENSE = None


def chamfer_loss(params):
    theta, M, X = params
    env = np.exp(M * np.abs(_T_DENSE)) * np.sin(0.3 * _T_DENSE)
    xr = _T_DENSE * np.cos(theta) - env * np.sin(theta) + X
    yr = 42 + _T_DENSE * np.sin(theta) + env * np.cos(theta)
    curve_pts = np.column_stack([xr, yr])
    tree = cKDTree(curve_pts)
    d, _ = tree.query(_PTS, k=1)
    return np.mean(d ** 2)


def run_global_search(pts):
    global _PTS, _T_DENSE
    _PTS = pts
    _T_DENSE = np.linspace(T_MIN, T_MAX, N_CURVE_SAMPLES_DE)
    t_dense = _T_DENSE
    bounds = [(0, np.deg2rad(50)), (-0.05, 0.05), (0, 100)]

    t0 = time.time()
    result = differential_evolution(
        chamfer_loss,
        bounds,
        seed=42,
        tol=1e-12,
        maxiter=300,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=True,
    )
    dt = time.time() - t0
    theta_de, M_de, X_de = result.x
    print(f"[Differential Evolution] time={dt:.1f}s  nfev={result.nfev}")
    print(
        f"[Differential Evolution] theta={np.rad2deg(theta_de):.5f} deg  "
        f"M={M_de:.6f}  X={X_de:.5f}  loss={result.fun:.8f}"
    )
    return theta_de, M_de, X_de, t_dense


def polish_with_joint_least_squares(pts, theta0, M0, X0, t_dense):
    n = len(pts)

    # Seed each point's t with the nearest curve sample under the DE result
    env = np.exp(M0 * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    xr = t_dense * np.cos(theta0) - env * np.sin(theta0) + X0
    yr = 42 + t_dense * np.sin(theta0) + env * np.cos(theta0)
    curve_pts = np.column_stack([xr, yr])
    tree = cKDTree(curve_pts)
    _, idx = tree.query(pts, k=1)
    t_init = t_dense[idx]

    def residuals(params):
        theta, M, X = params[:3]
        t = params[3:]
        env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
        xr = t * np.cos(theta) - env * np.sin(theta) + X
        yr = 42 + t * np.sin(theta) + env * np.cos(theta)
        return np.concatenate([xr - pts[:, 0], yr - pts[:, 1]])

    x0 = np.concatenate([[theta0, M0, X0], t_init])
    lb = np.concatenate([[0, -0.05, 0], np.full(n, T_MIN)])
    ub = np.concatenate([[np.deg2rad(50), 0.05, 100], np.full(n, T_MAX)])

    t0 = time.time()
    res = least_squares(
        residuals, x0, bounds=(lb, ub),
        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000,
    )
    dt = time.time() - t0

    theta_f, M_f, X_f = res.x[:3]
    resid = res.fun
    rms = np.sqrt(np.mean(resid ** 2))
    print(f"\n[Joint least-squares polish] time={dt:.1f}s  cost={res.cost:.3e}")
    print(f"[Joint least-squares polish] max|residual|={np.max(np.abs(resid)):.3e}  RMS={rms:.3e}")
    return theta_f, M_f, X_f, res


def main():
    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values

    print("=== Step 1: Global search (Differential Evolution + Chamfer loss) ===")
    theta_de, M_de, X_de, t_dense = run_global_search(pts)

    print("\n=== Step 2: Precision polish (joint least-squares over theta, M, X, t_1..t_1500) ===")
    theta_f, M_f, X_f, res = polish_with_joint_least_squares(pts, theta_de, M_de, X_de, t_dense)

    print("\n=== FINAL ANSWER ===")
    print(f"theta = {np.rad2deg(theta_f):.6f} deg  ({theta_f:.8f} rad)")
    print(f"M     = {M_f:.6f}")
    print(f"X     = {X_f:.6f}")

    with open("final_params.json", "w") as f:
        json.dump(
            {
                "theta_rad": theta_f,
                "theta_deg": float(np.rad2deg(theta_f)),
                "M": M_f,
                "X": X_f,
                "polish_max_abs_residual": float(np.max(np.abs(res.fun))),
                "polish_rms_residual": float(np.sqrt(np.mean(res.fun ** 2))),
            },
            f,
            indent=2,
        )
    print("\nSaved final_params.json")


if __name__ == "__main__":
    main()
