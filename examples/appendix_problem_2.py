#!/usr/bin/env python3
"""Appendix Problem 2: Van der Pol benchmark with trajectory, phase portrait, Newton telemetry, and residual trace."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numanalytica import BackwardEuler, NewtonRaphson, van_der_pol, van_der_pol_jacobian
from numanalytica.visualization.convergence_plot import plot_convergence_history

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)


def rhs(t, y):
    return van_der_pol(t, y, mu=1.0)


def jac(t, y):
    return van_der_pol_jacobian(t, y, mu=1.0)


if __name__ == "__main__":
    t0, tf = 0.0, 10.0
    y0 = np.array([2.0, 0.0])
    h = 0.10

    solver = BackwardEuler(rhs, jacobian=jac, verbose=False)
    start = __import__("time").perf_counter()
    result = solver.solve(t0=t0, tf=tf, y0=y0, h=h, newton_tol=1e-8, newton_maxiter=15)
    elapsed = __import__("time").perf_counter() - start

    counts = Counter(result.newton_iterations)
    print("Problem 2 Newton distribution")
    for k in sorted(counts):
        print(f"  {k} iters: {counts[k] / len(result.newton_iterations) * 100:.1f}%")
    print(f"  Max Newton / step: {max(result.newton_iterations)}")
    print(f"  Min Newton / step: {min(result.newton_iterations)}")
    print(f"  Final residual: {result.residual:.3e}")
    print(f"  Wall time: {elapsed:.6f} s")

    y1 = result.y[:, 0]
    y2 = result.y[:, 1]

    # Newton residual trace using the library's scalar Newton solver (educational transparency)
    def f(x):
        return x**3 - 2 * x - 5

    root_solver = NewtonRaphson(f, verbose=False)
    root_result = root_solver.solve(x0=2.0, tol=1e-10, maxiter=10)
    residuals = [entry["residual"] for entry in root_result.iteration_history]
    residuals = np.asarray(residuals, dtype=float)
    reference = residuals[0] * np.power(0.5, 2 * np.arange(len(residuals)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(result.t, y1, label="y1(t)", linewidth=2)
    axes[0, 0].plot(result.t, y2, label="y2(t)", linewidth=2)
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("State")
    axes[0, 0].set_title("Problem 2: trajectory")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(y1, y2, linewidth=2, color="C3")
    axes[0, 1].set_xlabel("y1")
    axes[0, 1].set_ylabel("y2")
    axes[0, 1].set_title("Problem 2: phase portrait")
    axes[0, 1].grid(alpha=0.3)

    colors = plt.cm.viridis(
        (np.array(result.newton_iterations) - min(result.newton_iterations))
        / max(1, max(result.newton_iterations) - min(result.newton_iterations))
    )
    axes[1, 0].scatter(
        np.arange(len(result.newton_iterations)), result.newton_iterations, c=colors, s=40
    )
    axes[1, 0].set_xlabel("Step index")
    axes[1, 0].set_ylabel("Newton iterations / step")
    axes[1, 0].set_title("Problem 2: Newton iterations per step")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].semilogy(np.arange(len(residuals)), residuals, "o-", label="Newton residual")
    axes[1, 1].semilogy(np.arange(len(reference)), reference, "k--", label="Quadratic reference")
    axes[1, 1].set_xlabel("Newton iteration")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Problem 2: residual convergence trace")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "appendix_problem_2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
