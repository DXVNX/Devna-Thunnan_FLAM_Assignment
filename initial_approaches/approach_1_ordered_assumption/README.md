# Approach 1 — "Assume the CSV rows are ordered by t"

## Why I chose this approach

This was the first thing worth checking before writing any optimizer. The
problem statement says points are given "for `6 < t < 60`", and the CSV
has exactly 1500 rows. The cheapest possible hypothesis is: row 0 is the
point at `t ≈ 6`, row 1499 is the point at `t ≈ 60`, and the 1500 points
are evenly spaced in between. If that were true, this stops being a hard
"unlabeled point cloud" problem and becomes a completely standard
`scipy.optimize.curve_fit` regression with a known `t` for every `(x, y)`.
It costs almost nothing to check, so it's the right first move even
though it's unlikely to be true.

## What I did

1. Built `t_assumed = linspace(6, 60, 1500)` and lined it up with the row
   order of `xy_data.csv`.
2. Plotted `x` and `y` against `t_assumed` (`ordering_sanity_check.png`).
   If the assumption were correct, both plots should look like smooth,
   gently curving lines (the underlying formula is smooth in `t`).
3. Ran `curve_fit` anyway against `t_assumed`, to put a number on how bad
   the assumption is instead of just eyeballing the plot.

## Result

The sanity-check plot is **not smooth** — it's a jagged, noisy scatter
that jumps up and down between consecutive rows with no visible trend.
That alone is enough to reject the assumption before even looking at the
fit.

The forced fit confirms it quantitatively:

```
theta = 29.5823 deg
M     = -0.05000   <- pinned at the boundary of the allowed range
X     = 55.0135
RMS residual = 16.0384
```

An RMS residual of ~16 units on data whose `x`/`y` values only span
roughly 50–110 and 46–70 is a very poor fit — the optimizer clamps `M` to
the edge of its allowed range purely to try (and fail) to explain noise
that has nothing to do with the true curve. Interestingly `theta` and `X`
land close to the eventual true values (≈30° and 55) — a hint that the
*shape* of the curve is not fundamentally different in scale from the
truth, but the mapping from row-index to `t` is simply wrong, so no
single `(θ, M, X)` can make it consistent point-by-point.

## Inference / why I moved on

**The rows are shuffled.** Whoever generated `xy_data.csv` computed 1500
`(x, y)` points from the formula and then randomized the row order before
saving — most likely to prevent exactly the shortcut I just tried. This
tells me something important about every approach from here on:

> I know each point lies *somewhere* on the curve, but I have no idea
> which `t` produced it. Any correct approach has to treat the data as an
> **unordered cloud of points** and can never rely on row position as a
> stand-in for `t`.

That single insight rules out plain regression entirely and pushed me
toward a correspondence-free way of scoring how well a candidate curve
matches the data (see Approach 2 and the Final Solution).
