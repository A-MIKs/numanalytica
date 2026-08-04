#!/usr/bin/env python3
"""Appendix Figure A.4: linear two-time-scale stiff benchmark used to illustrate stiff decay dynamics."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numanalytica import BackwardEuler

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# Linear two-time-scale stiffness benchmark:
# y' = A y,  with A = [[-1000, 0], [1, -1]], y(0) = [1, 1]^T.
# The eigenvalues are -1000 and -1, so the stiffness ratio is approximately 1000.
# The fast mode is y1 and the slow mode is y2.
A = np.array(
    [
        [-1000.0, 0.0],
        [1.0, -1.0],
    ]
)


def stiff_rhs(t, y):
    return A @ y


def stiff_jacobian(t, y):
    return A


if __name__ == "__main__":
    t0, tf = 0.0, 5.0
    y0 = np.array([1.0, 1.0])
    h = 0.5

    solver = BackwardEuler(stiff_rhs, jacobian=stiff_jacobian, verbose=False)
    start = time.perf_counter()
    result = solver.solve(t0=t0, tf=tf, y0=y0, h=h, newton_tol=1e-10, newton_maxiter=20)
    elapsed = time.perf_counter() - start

    y1 = result.y[:, 0]
    y2 = result.y[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(result.t, y1, label="y1", linewidth=2)
    axes[0].plot(result.t, y2, label="y2", linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("state")
    axes[0].set_title("Linear stiff benchmark: solution components")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(result.t, np.abs(y1), label="Fast mode", color="C0", linewidth=2)
    axes[1].semilogy(result.t, np.abs(y2), label="Slow mode", color="C1", linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|y(t)|")
    axes[1].set_title("Linear stiff benchmark: fast/slow time-scale separation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(result.t, np.abs(y1), color="C0", linewidth=2)
    axes[2].axhline(1e-15, color="k", linestyle="--", label="machine epsilon")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("|y1(t)|")
    axes[2].set_title("Linear stiff benchmark: fast component decay")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "appendix_linear_stiff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Linear stiff benchmark telemetry")
    print(f"  Steps: {len(result.step_sizes)}")
    print(f"  Max Newton / step: {max(result.newton_iterations)}")
    print(f"  Min Newton / step: {min(result.newton_iterations)}")
    print(f"  Final residual: {result.residual:.3e}")
    print(f"  Wall time: {elapsed:.6f} s")
    print("  Note: for this linear system, Newton converges in a single correction under the solver's stopping criterion.")
