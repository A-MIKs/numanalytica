#!/usr/bin/env python3
"""Appendix Figure A.3: stiffness comparison for Van der Pol with mu = 1, 10, 100."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numanalytica import BackwardEuler, van_der_pol, van_der_pol_jacobian

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)


PARAMS = [
    (1.0, 0.10),
    (10.0, 0.02),
    (100.0, 0.02),
]


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for idx, (mu, h) in enumerate(PARAMS):
        rhs = lambda t, y, mu=mu: van_der_pol(t, y, mu=mu)
        jac = lambda t, y, mu=mu: van_der_pol_jacobian(t, y, mu=mu)

        solver = BackwardEuler(rhs, jacobian=jac, verbose=False)
        start = __import__("time").perf_counter()
        result = solver.solve(
            t0=0.0, tf=10.0, y0=np.array([2.0, 0.0]), h=h, newton_tol=1e-8, newton_maxiter=15
        )
        elapsed = __import__("time").perf_counter() - start

        y1 = result.y[:, 0]
        y2 = result.y[:, 1]

        axes[idx, 0].plot(result.t, y1, label="y1", linewidth=2)
        axes[idx, 0].plot(result.t, y2, label="y2", linewidth=2)
        axes[idx, 0].set_title(f"μ = {mu}; h = {h}")
        axes[idx, 0].set_xlabel("t")
        axes[idx, 0].set_ylabel("state")
        axes[idx, 0].legend()
        axes[idx, 0].grid(alpha=0.3)

        axes[idx, 1].plot(y1, y2, linewidth=2, color="C3")
        axes[idx, 1].set_xlabel("y1")
        axes[idx, 1].set_ylabel("y2")
        axes[idx, 1].set_title("phase portrait")
        axes[idx, 1].grid(alpha=0.3)

        print(
            f"mu={mu}, h={h}: avg Newton/step={np.mean(result.newton_iterations):.2f}, final residual={result.residual:.3e}, wall={elapsed:.6f}s"
        )

    fig.tight_layout()
    fig.savefig(OUTDIR / "appendix_stiff_van_der_pol.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
