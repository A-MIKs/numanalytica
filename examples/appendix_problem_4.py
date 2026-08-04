#!/usr/bin/env python3
"""Appendix Problem 4: convergence proof with log-log error plot and exact-solution overlay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numanalytica import BackwardEuler
from numanalytica.visualization.convergence_plot import (
    plot_error_vs_stepsize,
    plot_ode_solution,
)

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)


def rhs(t, y):
    return -y


def exact_solution(t):
    return np.exp(-t)


if __name__ == "__main__":
    t0, tf = 0.0, 1.0
    y0 = np.array([1.0])
    step_sizes = np.array([0.5, 0.25, 0.125, 0.0625])
    errors = []

    for h in step_sizes:
        solver = BackwardEuler(rhs, verbose=False)
        result = solver.solve(t0=t0, tf=tf, y0=y0, h=h, newton_tol=1e-10, newton_maxiter=20)
        exact = exact_solution(result.t)
        errors.append(float(np.max(np.abs(result.y[:, 0] - exact))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_error_vs_stepsize(
        np.asarray(step_sizes),
        np.asarray(errors),
        order=1,
        title="Problem 4: log-log convergence proof",
        ax=axes[0],
    )

    h = 0.1
    solver = BackwardEuler(rhs, verbose=False)
    result = solver.solve(t0=t0, tf=tf, y0=y0, h=h, newton_tol=1e-10, newton_maxiter=20)
    exact = exact_solution(result.t)
    plot_ode_solution(
        result.t,
        result.y[:, 0],
        t_exact=result.t,
        y_exact=exact,
        title="Problem 4: solution versus exact",
        ax=axes[1],
    )

    fig.tight_layout()
    fig.savefig(OUTDIR / "appendix_problem_4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Problem 4 convergence telemetry")
    print(f"  Step sizes: {step_sizes}")
    print(f"  Errors: {errors}")
    print("Problem 4 convergence telemetry")
    print(f"  Step sizes: {step_sizes}")
    print(f"  Errors: {errors}")
