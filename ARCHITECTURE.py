"""
NumAnalytica: A Pedagogical Framework for Backward Differentiation Formulas

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

NumAnalytica is a "glass-box" Python library designed to bridge the gap between
theoretical numerical analysis and practical software engineering. Unlike
commercial black-box solvers (SciPy, MATLAB), it exposes every algorithmic
step for educational transparency.

================================================================================
MODULE STRUCTURE (src/numanalytica/)
================================================================================

1. CORE FOUNDATION (core/)
   ├── exceptions.py       - Custom exception hierarchy for all error conditions
   ├── results.py          - Unified output containers (SolverResult, RootResult, IntegrationResult)
   ├── logger.py           - IterationLogger: The pedagogical engine
   └── base_solver.py      - BaseSolver: Abstract interface for all solvers

2. NUMERICAL DIFFERENTIATION (differentiation/)
   ├── complex_step.py     - Complex Step Differentiation (YOUR KEY INNOVATION)
   │                         • Derivative with machine precision (h=1e-20)
   │                         • Jacobian computation
   │                         • Gradient computation
   │
   ├── finite_diff.py      - Standard finite differences for comparison
   │                         • Forward, backward, centered methods
   │                         • Shows why CSD is better
   │
   └── jacobian.py         - JacobianComputer abstraction, linear solvers

3. ROOT-FINDING METHODS (roots/) - ALL NOW OOP with BaseSolver
   ├── newton_raphson.py   - Newton-Raphson (scalar + system)
   │                         • Automatic differentiation support
   │                         • Uses Complex Step Jacobian for systems
   │
   ├── bisection.py        - Bisection (robust but slow)
   ├── secant.py           - Secant (derivative-free, faster than bisection)
   ├── fixed_point.py      - Fixed-point iteration (x = g(x))
   ├── false_position.py   - Regula Falsi (interpolation-based bracketing)
   ├── muller.py           - Müller's method (quadratic interpolation)
   └── utils.py            - Bracketing utilities

4. ODE INTEGRATION (ode/)
   ├── base_integrator.py  - BaseIntegrator: Common interface for all methods
   │
   ├── explicit/
   │   └── euler.py        - Forward Euler (explicit baseline)
   │                         • Shows why explicit fails on stiff problems
   │                         • O(h) error per step
   │
   └── implicit/
       └── backward_euler.py - Backward Euler (BDF Order 1, A-STABLE)
                               • THE CORE OF YOUR THESIS
                               • Solves implicit equation via Newton-Raphson
                               • Uses Complex Step Jacobian
                               • Handles stiff equations with large steps
                               • Fully A-stable (stability region = left half-plane)

5. STABILITY ANALYSIS (stability/)
   └── region_plotter.py
       ├── StabilityRegion class
       │   • Computes stability regions in complex plane
       │   • Visualizes with matplotlib
       │   • Tests for A-stability
       │
       └── Pre-built methods:
           • forward_euler (limited disk)
           • backward_euler (A-STABLE ✓)
           • BDF2 (higher-order)
           • RK4 (explicit, limited)

6. BENCHMARKS (benchmarks/)
   └── __init__.py
       ├── Van der Pol oscillator (μ=1 and μ=100)
       │   • Perfect for demonstrating stiffness effects
       │   • Shows Backward Euler advantages
       │
       ├── BenchmarkProblem class
       └── BENCHMARK_LIBRARY

7. VISUALIZATION (visualization/)
   └── Placeholder for future plotting utilities

================================================================================
KEY FEATURES
================================================================================

✓ PEDAGOGICAL TRANSPARENCY
  • IterationLogger captures every step
  • Formatted iteration tables for analysis
  • step-by-step RHS and Jacobian evaluations
  • Newton iteration counts per step

✓ COMPLEX STEP DIFFERENTIATION (Your Innovation)
  • f'(x) ≈ Im(f(x+ih)) / h with h=1e-20
  • Machine precision, no round-off errors
  • Avoids subtractive cancellation
  • Critical for accurate Jacobians in Newton-Raphson

✓ IMPLICIT ODE SOLVER (Backward Euler)
  • Solves y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1})
  • Via Newton-Raphson at each step
  • Uses Complex Step Jacobian
  • A-stable: handles stiff problems

✓ UNIFIED RESULT OBJECTS
  • All solvers return SolverResult subclasses
  • Consistent API across all methods
  • Iteration history always available

✓ OOP ARCHITECTURE
  • All solvers inherit from BaseSolver
  • Type hints throughout
  • Comprehensive docstrings + examples
  • Easy to extend with new methods

================================================================================
MATHEMATICAL JUSTIFICATION (Chapter 3 of Thesis)
================================================================================

3.1 NUMERICAL DIFFERENTIATION
    - Complex Step Differentiation (Martins et al. 2003)
    - Avoids finite difference round-off dilemma
    - Achieves machine precision with tiny h

3.2 ROOT-FINDING WITH DIFFERENTIATION
    - Newton-Raphson iteration
    - Convergence rate: quadratic
    - Jacobian via Complex Step

3.3 IMPLICIT ODE SOLVERS (BDF)
    - Backward Euler: y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1})
    - Implicit equation solved via Newton-Raphson
    - Stability region analysis in complex plane

3.4 A-STABILITY ANALYSIS
    - Dahlquist theory
    - Stability function ρ(z)
    - Backward Euler: A-STABLE (Re(z) < 0 fully covered)

================================================================================
USAGE EXAMPLES
================================================================================

# 1. ROOT-FINDING WITH AUTO-DIFFERENTIATION
from numanalytica import NewtonRaphson

def f(x):
    return x**3 - 2*x - 5

solver = NewtonRaphson(f)  # Automatic Complex Step differentiation!
result = solver.solve(x0=2.0)
print(result)  # Formatted output with convergence details
print(solver.get_iteration_table())  # See every iteration


# 2. SYSTEM OF EQUATIONS
from numanalytica import NewtonRaphsonSystem

def F(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

solver = NewtonRaphsonSystem(F)
result = solver.solve(x0=np.array([0.5, 0.5]))


# 3. ODE INTEGRATION (STIFF PROBLEM)
from numanalytica import BackwardEuler, van_der_pol, van_der_pol_jacobian

solver = BackwardEuler(
    f=lambda t, y: van_der_pol(t, y, mu=1.0),
    jacobian=lambda t, y: van_der_pol_jacobian(t, y, mu=1.0)
)
result = solver.solve(t0=0, tf=10, y0=np.array([2.0, 0.0]), h=0.25)
print(result)  # Full diagnostics
print(f"Newton iterations per step: {np.mean(result.newton_iterations):.2f}")


# 4. STABILITY REGION VISUALIZATION
from numanalytica import StabilityRegion, stability_backward_euler

region = StabilityRegion(stability_backward_euler, "Backward Euler")
region.plot_region()  # Beautiful matplotlib visualization


================================================================================
THESIS INTEGRATION (Chapter 4)
================================================================================

Section 4.1: Implementation
  - Show actual Python code
  - Reference specific modules/classes
  - Explain design choices

Section 4.2: Results on Van der Pol
  - Integration from t=0 to t=10
  - Step size h=0.25 (fixed for demo)
  - Convergence of Newton iterations
  - Final solution trajectory plot

Section 4.3: Stability Region Plots
  - Forward Euler vs Backward Euler side-by-side
  - Visual proof of A-stability
  - Theoretical explanation

Section 4.4: Performance Benchmarking
  - Compare Newton iteration counts
  - RHS and Jacobian evaluations
  - Computation time
  - Accuracy metrics

================================================================================
NEXT STEPS FOR COMPLETION
================================================================================

PRIORITY 1: Fine-tune Backward Euler
  - Debug Newton solver convergence on Van der Pol
  - May need step size adaptation
  - May need better initial guesses

PRIORITY 2: Implement BDF2 (Second-order)
  - More accurate than Backward Euler
  - Still A-stable
  - Uses 2-step history

PRIORITY 3: Comprehensive Tests
  - Unit tests for all modules
  - Integration tests for entire workflow
  - Regression tests for known solutions

PRIORITY 4: Documentation
  - Jupyter notebooks (1 per major feature)
  - Thesis Chapter 4 write-up with actual results
  - API reference documentation

================================================================================
IMPORTANT NOTES FOR YOUR SUPERVISOR
================================================================================

1. This is a TEACHING TOOL, not a production library
   - Emphasize pedagogical transparency
   - Show internal iterations/errors
   - Every step is traceable

2. Complex Step Differentiation is YOUR innovation
   - Shows machine precision
   - Eliminates subtractive cancellation
   - Critical for Newton-Raphson Jacobians

3. Backward Euler demonstrates BDF principal
   - A-stable (unlike Forward Euler)
   - Handles stiffness elegantly
   - Newton-Raphson solves implicit equation

4. Stability region visualization is powerful
   - Visual proof of theoretical stability
   - Directly connected to complex analysis
   - Shows why explicit methods fail on stiff problems

================================================================================
"""

print(__doc__)
