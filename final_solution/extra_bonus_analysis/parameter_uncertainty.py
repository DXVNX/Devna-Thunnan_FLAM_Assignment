"""
Bonus / extra math: parameter uncertainty from the least-squares Jacobian
===========================================================================
The Final Solution reports point estimates for (theta, M, X). This script
adds formal uncertainty bounds using standard nonlinear least-squares
theory: at the optimum, the covariance of the fitted parameters can be
approximated as

    Cov(params) ~= residual_variance * (J^T J)^-1

where J is the Jacobian of the residual vector with respect to the
parameters at the optimum, and residual_variance is the sample variance
of the residuals (since the true per-point noise level isn't given, it is
estimated from the residuals themselves).

This reuses the same joint fit as final_solution.py (theta, M, X, and one
t_i per data point) so the Jacobian block for (theta, M, X) already
accounts for every other point's optimal t_i.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

DATA_PATH = "../../data/xy_data.csv"
PARAMS_PATH = "../final_params.json"
T_MIN, T_MAX = 6, 60


def curve_xy(theta, M, X, t):
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - env * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + env * np.cos(theta)
    return x, y


def main():
    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values
    n = len(pts)

    with open(PARAMS_PATH) as f:
        p = json.load(f)
    theta0, M0, X0 = p["theta_rad"], p["M"], p["X"]

    # seed t_i via nearest neighbor at the already-known optimum
    t_dense = np.linspace(T_MIN, T_MAX, 4000)
    x_dense, y_dense = curve_xy(theta0, M0, X0, t_dense)
    tree = cKDTree(np.column_stack([x_dense, y_dense]))
    _, idx = tree.query(pts, k=1)
    t_init = t_dense[idx]

    def residuals(params):
        theta, M, X = params[:3]
        t = params[3:]
        x, y = curve_xy(theta, M, X, t)
        return np.concatenate([x - pts[:, 0], y - pts[:, 1]])

    x0 = np.concatenate([[theta0, M0, X0], t_init])
    lb = np.concatenate([[0, -0.05, 0], np.full(n, T_MIN)])
    ub = np.concatenate([[np.deg2rad(50), 0.05, 100], np.full(n, T_MAX)])

    res = least_squares(residuals, x0, bounds=(lb, ub),
                         xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)

    resid = res.fun
    dof = max(1, len(resid) - len(res.x))  # degrees of freedom
    resid_var = np.sum(resid ** 2) / dof

    J = res.jac
    JTJ = J.T @ J
    try:
        cov_full = resid_var * np.linalg.inv(JTJ)
        cov_theta_M_X = cov_full[:3, :3]
        se = np.sqrt(np.diag(cov_theta_M_X))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan, np.nan])

    theta_f, M_f, X_f = res.x[:3]
    z95 = 1.96

    print("--- Parameter estimates with 95% confidence intervals ---")
    print(f"theta = {np.rad2deg(theta_f):.6f} deg  +/- {np.rad2deg(se[0])*z95:.2e} deg (95% CI)")
    print(f"M     = {M_f:.6f}       +/- {se[1]*z95:.2e}     (95% CI)")
    print(f"X     = {X_f:.6f}      +/- {se[2]*z95:.2e}     (95% CI)")
    print(
        f"\nResidual variance (proxy for per-point noise^2): {resid_var:.3e}"
    )
    print(
        "\nNote: the residuals here are already at numerical-precision "
        "level (~1e-6), so these confidence intervals are extremely tight "
        "-- consistent with the data having been generated from these "
        "exact parameter values with no added noise."
    )


if __name__ == "__main__":
    main()
