# NumAnalytica: A Pedagogical ODE Solver Framework

## Overview

**NumAnalytica** is a pedagogical Python library for the numerical solution of stiff ordinary differential equations. The project focuses on transparent, educational implementations of implicit methods, especially Backward Differentiation Formulas (BDFs), and it exposes the internal algorithmic steps so that the numerical behaviour can be inspected directly.

This repository forms the software component of the final-year B.Sc. thesis:
> *Implementation and Stability Analysis of Backward Differentiation Formulas: The NumAnalytica Library Approach*

### Key Feature: Complex Step Differentiation

The library uses **Complex Step Differentiation (CSD)** to compute Jacobians with near-machine-precision accuracy. This reduces the subtractive-cancellation error that commonly affects finite-difference approximations and improves the reliability of Newton iterations in implicit solvers.

---

## Architecture

```
src/numanalytica/
├── core/                    # Foundation: exceptions, results, logger, base class
├── differentiation/         # Complex Step & Finite Differences
├── roots/                   # 6 root-finding algorithms (all OOP)
├── ode/
│   ├── explicit/           # Forward Euler (baseline)
│   └── implicit/           # Backward Euler (A-stable BDF)
├── stability/              # A-Stability visualization
├── benchmarks/             # Van der Pol oscillator
└── visualization/          # Plotting utilities
```

---

## Installation

```bash
# Install in development mode (from repo root)
pip install -e .
```

### Dependencies
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.5

---

## Quick Start

### 1. Root-Finding (Automatic Differentiation)

```python
from numanalytica import NewtonRaphson

def f(x):
    return x**3 - 2*x - 5

solver = NewtonRaphson(f)  # Auto-uses Complex Step!
result = solver.solve(x0=2.0)

print(result)  # Formatted output
print(solver.get_iteration_table())  # See every iteration
```

**Output:**
```
============================================================
  Root Finding Result
============================================================
Status:            ✓ Converged
Root:              2.094551481542327e+00
Residual:          8.88e-16
Iterations:        5
Computation Time:  0.0007 s
============================================================

 Iteration History Table
================================================================================
  Iter |     Residual |       x
-------+--------------+--------
     0 |     1.00e+00 |       2
     1 |     6.10e-02 |     2.1
     2 |     1.86e-04 | 2.09457
     3 |     1.74e-09 | 2.09455
     4 |     8.88e-16 | 2.09455
================================================================================
```

### 2. ODE Integration (Implicit Solver)

```python
import numpy as np
from numanalytica import BackwardEuler, van_der_pol, van_der_pol_jacobian

# Define ODE: dy/dt = f(t, y)
f_vdp = lambda t, y: van_der_pol(t, y, mu=1.0)
J_vdp = lambda t, y: van_der_pol_jacobian(t, y, mu=1.0)

# Create implicit solver (A-stable)
solver = BackwardEuler(f_vdp, jacobian=J_vdp)

# Solve from t=0 to t=10
result = solver.solve(
    t0=0, tf=10,
    y0=np.array([2.0, 0.0]),
    h=0.25
)

print(result)
print(f"Newton iters/step: {np.mean(result.newton_iterations):.2f}")
```

### 3. Stability Region Visualization

```python
from numanalytica import stability_backward_euler, StabilityRegion

region = StabilityRegion(stability_backward_euler, "Backward Euler")
fig, ax = region.plot_region()
plt.show()  # Beautiful complex-plane visualization
```

### 4. Example Appendix Benchmarks

The repository includes runnable example scripts under `examples/` that reproduce the benchmark figures used in the thesis appendix. These examples are intended to support validation and reproducibility:

- `appendix_linear_stiff.py` — linear two-time-scale stiff benchmark
- `appendix_problem_1.py` — scalar decay benchmark
- `appendix_problem_2.py` — nonlinear Newton / phase-portrait experiment
- `appendix_problem_4.py` — convergence-order verification
- `appendix_stiff_van_der_pol.py` — stiffness comparison across values of `mu`

These scripts generate the corresponding figures in `examples/figures/` and can be run directly from the repository root.

---

## Core Components

### 🔷 Complex Step Differentiation (differentiation/complex_step.py)

```python
from numanalytica import complex_step_derivative

def f(x):
    return np.sin(x) * np.exp(-x**2/2)

# Compute f'(x) with machine precision
fprime = complex_step_derivative(f, x=1.5, h=1e-20)
# Error ~ 10^-16 (much better than finite differences!)
```

**Why it matters:**
- Avoids the round-off vs. truncation error balance problem
- Jacobians for Newton-Raphson are accurate to machine precision
- Enables fast, robust convergence of implicit solvers

### 🔷 OOP Root-Finding (roots/)

All methods extend `BaseSolver`:
- `NewtonRaphson` - Fast, needs derivative
- `Bisection` - Slow, robust
- `Secant` - Fast, derivative-free
- `FalsePosition` - Interpolation-based
- `FixedPoint` - Iteration x = g(x)
- `Muller` - Quadratic interpolation

### 🔷 Implicit ODE Solver (ode/implicit/backward_euler.py)

The **heart**:

1. **Backward Euler step:**
   $$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

2. **Implicit equation:** Solved using `NewtonRaphsonSystem`
3. **Jacobian:** Computed via Complex Step Differentiation
4. **A-Stability:** Proven stable for all Re(z) < 0

### 🔷 A-Stability Analysis (stability/region_plotter.py)

```python
from numanalytica import StabilityComparison, STABILITY_LIBRARY

# Side-by-side comparison
comparison = StabilityComparison({
    "Forward Euler": STABILITY_LIBRARY["Forward Euler"],
    "Backward Euler": STABILITY_LIBRARY["Backward Euler"],
})
comparison.plot_comparison()
```

---

## Pedagogical Features

### 1. Iteration Logging

Every solver logs every iteration:
```python
result = solver.solve(...)
print(solver.get_iteration_table())  # Formatted table
```

### 2. Unified Result Objects

All solvers return `SolverResult` or subclasses:
```python
result.converged        # bool
result.iterations       # int
result.residual        # float
result.solution        # np.ndarray or float
result.elapsed_time    # float
result.iteration_history  # List[Dict]
```

### 3. Verbose Output

Solvers print headers, progress, and detailed summaries:
```
======================================================================
  Newton-Raphson
======================================================================
  Iter   0: x=2.000000 | residual=1.00e+00 | |step|=1.00e+00
  Iter   1: x=2.100000 | residual=6.10e-02 | |step|=1.00e-01
  ...
```

---

## Comparison with Production Solvers

| Feature | NumAnalytica | SciPy odeint |
|---------|--------------|--------------|
| **Iteration details** | ✓ Full transparency | ✗ Black-box |
| **Stability regions** | ✓ Visualized | ✗ Not exposed |
| **Method education** | ✓ Explicit steps | ✗ Hidden |
| **Jacobian control** | ✓ CSD available | Limited |
| **Speed** | Pedagogical | Optimized |
| **Purpose** | Teaching | Production |

---

## Project Structure

### Repository Layout
```
numanalytica/
├── pyproject.toml          # Package metadata and build configuration
├── setup.py                # Backward-compatibility shim
├── src/
│   └── numanalytica/       # Source package
├── examples/               # Runnable appendix and benchmark examples
├── docs/                   # Documentation resources
├── tests/                  # Test suite
├── README.md               # Project homepage
└── HISTORY.md              # Release history
```

---

## Testing

```bash
# Run the test suite
pytest tests/ -v

# Run the demo script
python demo.py
```

---

## Example: Full Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from numanalytica import (
    BackwardEuler,
    van_der_pol,
    van_der_pol_jacobian,
    StabilityRegion,
    stability_backward_euler,
)

# 1. Setup ODE
f = lambda t, y: van_der_pol(t, y, mu=1.0)
J = lambda t, y: van_der_pol_jacobian(t, y, mu=1.0)

# 2. Solve (A-stable method!)
solver = BackwardEuler(f, jacobian=J, verbose=True)
result = solver.solve(
    t0=0, tf=10,
    y0=np.array([2.0, 0.0]),
    h=0.25
)

# 3. Analyze
print(f"Converged: {result.converged}")
print(f"Steps: {len(result.t)}")
print(f"Newton iters per step: {np.mean(result.newton_iterations):.2f}")

# 4. Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Solution trajectory
ax1.plot(result.t, result.y[:, 0], 'b-o', label='y₁=position')
ax1.plot(result.t, result.y[:, 1], 'r-s', label='y₂=velocity')
ax1.legend()
ax1.set_xlabel('Time')
ax1.set_ylabel('Solution')
ax1.set_title('Van der Pol Integration (Backward Euler)')

# Stability region
region = StabilityRegion(stability_backward_euler, "Backward Euler")
region.plot_region(ax=ax2)

plt.tight_layout()
plt.show()

# 5. Export iteration table for thesis
table_latex = solver.get_iteration_table()
print(table_latex)  # Paste into thesis!
```

---

## Development Status

- ✅ Core numerical infrastructure
- ✅ Differentiation utilities and Jacobian tools
- ✅ Root-finding methods
- ✅ ODE solvers, including implicit Backward Euler
- ✅ Stability-region analysis and plotting
- ✅ Benchmark and appendix example scripts
- ✅ Publication-facing documentation updates
- ⏳ Expanded formal test coverage and further documentation work

---

## References

### Key Papers
- **Martins, J.R.R.A., Sturdza, P., & Alonso, J.J. (2003).** "The complex-step derivative approximation." *ACM Transactions on Mathematical Software*, 29(3), 245-262.
- **Dahlquist, G. (1963).** "A special stability problem for linear multistep methods." *BIT Numerical Mathematics*, 3(1), 27-43.
- **Butcher, J.C. (2016).** "Numerical methods for ordinary differential equations" (3rd ed.). John Wiley & Sons.

### Online Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

## License

MIT License - See LICENSE file

---

## Author

**Kola-Ilugbo, Ayomikun Fawaz**  
B.Sc. (Hons) Industrial Mathematics (Computer Option)  
University of Lagos, January 2026  
Supervisor: Dr. Hamzat, Jamiu O.

---

## Contact

📧 ayomikun.kolailugbo@gmail.com
🔗 [GitHub](https://github.com/A-MIKs/numanalytica)

---

**Last Updated:** September 2026
**Status:** Thesis-ready demonstration repository
