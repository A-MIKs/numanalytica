#!/usr/bin/env python3
"""Appendix Problem 1: scalar decay benchmark with log-scale trajectory and error bars."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numanalytica import BackwardEuler

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)


def rhs(t, y):
    return -5.0 * y


def exact_solution(t):
    return np.exp(-5.0 * t)


if __name__ == "__main__":
    t0, tf = 0.0, 1.0
    y0 = np.array([1.0])
    h = 0.05

    solver = BackwardEuler(rhs, verbose=False)
    start = __import__("time").perf_counter()
    result = solver.solve(t0=t0, tf=tf, y0=y0, h=h, newton_tol=1e-10, newton_maxiter=20)
    elapsed = __import__("time").perf_counter() - start

    t = result.t
    y = result.y[:, 0]
    y_exact = exact_solution(t)
    errors = np.abs(y - y_exact)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, y, label="NumAnalytica", linewidth=2, color="C0")
    axes[0].plot(t, y_exact, "k--", label="Exact", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y(t)")
    axes[0].set_title("Problem 1: log-scale trajectory")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(t, errors, yerr=0.5 * errors, fmt="o", color="C1", capsize=3)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|y_n - y_exact(t_n)|")
    axes[1].set_title("Per-step absolute error")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "appendix_problem_1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Problem 1 telemetry")
    print(f"  Steps: {len(result.step_sizes)}")
    print(f"  Max Newton / step: {max(result.newton_iterations)}")
    print(f"  Min Newton / step: {min(result.newton_iterations)}")
    print(f"  Final residual: {result.residual:.3e}")
    print(f"  Wall time: {elapsed:.6f} s")
