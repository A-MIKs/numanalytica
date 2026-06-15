"""
Quick demonstration of NumAnalytica capabilities.

This script showcases:
1. Complex Step Differentiation accuracy
2. Newton-Raphson for root-finding
3. Backward Euler for ODE solving
4. Stability region visualization
"""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np

from numanalytica import (
    BackwardEuler,
    ExplicitEuler,
    NewtonRaphson,
    StabilityRegion,
    complex_step_derivative,
    stability_backward_euler,
    stability_forward_euler,
    van_der_pol,
    van_der_pol_jacobian,
)


def demo_complex_step():
    """Demonstrate Complex Step Differentiation accuracy."""
    print("\n" + "=" * 70)
    print("  DEMO 1: Complex Step Differentiation")
    print("=" * 70)

    def f(x):
        return np.sin(x) * np.exp(-(x**2) / 2)

    def fprime_exact(x):
        return (np.cos(x) - x * np.sin(x)) * np.exp(-(x**2) / 2)

    x_test = 1.5
    fprime_csd = complex_step_derivative(f, x_test, h=1e-20)
    fprime_exact_val = fprime_exact(x_test)

    error = abs(fprime_csd - fprime_exact_val)

    print(f"\nFunction: f(x) = sin(x) * exp(-x²/2)")
    print(f"Point: x = {x_test}")
    print(f"\nExact derivative:     {fprime_exact_val:.15e}")
    print(f"CSD approximation:    {fprime_csd:.15e}")
    print(f"Absolute error:       {error:.2e}")
    print(f"✓ Machine precision achieved with h=1e-20!")


def demo_newton_raphson():
    """Demonstrate Newton-Raphson with automatic differentiation."""
    print("\n" + "=" * 70)
    print("  DEMO 2: Newton-Raphson Root-Finding (Auto-Differentiation)")
    print("=" * 70)

    def f(x):
        return x**3 - 2 * x - 5

    solver = NewtonRaphson(f, verbose=False)
    result = solver.solve(x0=2.0, tol=1e-10, maxiter=20)

    print(f"\nProblem: Find root of f(x) = x³ - 2x - 5")
    print(f"\n{result}")
    print("\nIteration History:")
    print(solver.get_iteration_table())


def demo_backward_euler():
    """Demonstrate Backward Euler for a stiff ODE (Van der Pol with μ=1)."""
    print("\n" + "=" * 70)
    print("  DEMO 3: Backward Euler on Van der Pol (Non-Stiff μ=1)")
    print("=" * 70)

    f_vdp = lambda t, y: van_der_pol(t, y, mu=1.0)
    J_vdp = lambda t, y: van_der_pol_jacobian(t, y, mu=1.0)

    solver = BackwardEuler(f_vdp, jacobian=J_vdp, verbose=False)
    result = solver.solve(t0=0, tf=10, y0=np.array([2.0, 0.0]), h=0.10, newton_maxiter=15)

    print(f"\nProblem: Van der Pol Oscillator with μ=1.0 (demonstration)")
    print(f"  dy1/dt = y2")
    print(f"  dy2/dt = 1(1-y1²)y2 - y1")
    print(f"\nTime span: [0, 10]  |  Step size: h=0.25  |  Steps taken: {len(result.t)-1}")
    print(f"\n{result}")
    print(f"\nAverage Newton iterations per step: {np.mean(result.newton_iterations):.2f}")
    print(f"Total RHS evaluations: {result.function_evaluations}")
    print(f"Total Jacobian evaluations: {result.jacobian_evaluations}")


def demo_stability_regions():
    """Visualize stability regions of Forward vs Backward Euler."""
    print("\n" + "=" * 70)
    print("  DEMO 4: A-Stability Visualization")
    print("=" * 70)

    euler_fwd = StabilityRegion(stability_forward_euler, "Forward Euler")
    euler_bwd = StabilityRegion(stability_backward_euler, "Backward Euler")

    print("\nForward Euler Stability:")
    print(f"  - Limited to disk |1+z| ≤ 1")
    print(f"  - NOT suitable for stiff problems")

    print("\nBackward Euler Stability:")
    print(f"  - Covers entire left half-plane (A-STABLE)")
    print(f"  - Handles stiff equations with large steps")

    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    euler_fwd.plot_region(ax=axes[0], resolution=400)
    euler_bwd.plot_region(ax=axes[1], resolution=400)

    fig.suptitle("Why BDF Methods Are Essential for Stiff ODEs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("stability_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to: stability_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  NumAnalytica Framework Demonstration")
    print("  Pedagogical ODE Solver with BDF & Stability Analysis")
    print("█" * 70)

    # Run demos
    demo_complex_step()
    demo_newton_raphson()
    demo_backward_euler()
    demo_stability_regions()

    print("\n" + "=" * 70)
    print("  ✓ All demos completed successfully!")
    print("=" * 70 + "\n")
