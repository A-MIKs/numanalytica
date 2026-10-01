# NumAnalytica: A Pedagogical ODE Solver Framework

## Overview

**NumAnalytica** is a small Python library for solving ordinary differential equations and related numerical-analysis problems with an emphasis on transparency. The project is intended to make the internal steps of the methods visible so that students, researchers, and prototyping workflows can inspect what is happening rather than rely on a black box.

This repository is the software component of the final-year B.Sc. thesis:
> *Implementation and Stability Analysis of Backward Differentiation Formulas: The NumAnalytica Library Approach*

### Key Feature: Complex Step Differentiation

The library uses **Complex Step Differentiation (CSD)** to compute Jacobians. For suitable functions that preserve the complex perturbation, and with appropriate step sizes, CSD can achieve near-machine-precision accuracy. It avoids the subtractive cancellation that often affects finite-difference approximations and can support Newton-type iterations in implicit solver workflows.

---

## Current verified status

The project currently has a compact regression suite covering the core solver workflows, solver edge cases, and appendix reproducibility checks. The emphasis is on correctness, transparency, and reproducibility rather than broad production coverage.

Verified in the current workspace with:

```bash
python verify_examples.py
pytest -q
```

Result:

```text
All 5 example scripts completed successfully.
10 tests passed.
```

The validated scenarios include scalar root finding, implicit Euler integration, output time selection, appendix benchmark regeneration, and key solver error handling. The library remains intentionally focused on a pedagogical numerical-analysis stack rather than a full-featured commercial solver package.

---

## Thesis-ready scope and limitations

This release is best understood as a thesis-ready numerical-analysis artifact rather than a general-purpose production library.

The current implementation is intentionally scoped to the methods and ideas directly supported by the project narrative:

- scalar and system Newton-type solvers
- bisection, secant, false position, fixed-point, and Muller methods
- backward Euler / implicit integration workflows
- complex-step differentiation for accurate derivative evaluation
- stability-region and benchmark visualizations

The project is not yet a broad-purpose scientific software platform. It does not aim to match the breadth, ecosystem integration, or numerical robustness guarantees of mature libraries such as SciPy, PETSc, or Julia’s DifferentialEquations ecosystem. Instead, the goal is to provide a transparent, educational, and reproducible implementation of the methods presented in the thesis.

For the thesis release, this is a strength: the code is explicit, inspectable, and methodologically aligned with the work being demonstrated.

---

## Architecture

```
src/numanalytica/
├── core/                    # Foundation: exceptions, results, logger, and base classes
├── differentiation/         # Complex-step and finite-difference utilities
├── roots/                   # Root-finding algorithms
├── ode/
│   ├── explicit/           # Explicit integration methods
│   └── implicit/           # Implicit integration methods, including backward Euler
├── stability/              # Stability-region analysis and visualizations
├── benchmarks/             # Benchmark ODE models and problem definitions
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

### 1. Root-Finding (Complex-Step Numerical Differentiation)

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

### Reproducibility and Appendix Validation

To reproduce the thesis appendix figures and confirm the example scripts still pass in a fresh environment, run:

```bash
python verify_examples.py
```

This command executes each appendix script in `examples/` and fails immediately if any benchmark or figure-generation routine stops working. This is the most direct researcher-facing check before assessing the numerical claims of the work.

---

## Core Components

### 🔷 Complex Step Differentiation (differentiation/complex_step.py)

Complex Step Differentiation (CSD) is a numerical differentiation technique,
not automatic differentiation. For suitable functions and step sizes, it can
achieve near-machine-precision derivatives by avoiding subtractive cancellation.
The function must preserve the complex perturbation and behave analytically near
the evaluation point. A real cast, non-analytic operation, or other operation
that discards or alters the imaginary component can produce incorrect results.

```python
from numanalytica import complex_step_derivative

def f(x):
    return np.sin(x) * np.exp(-x**2/2)

# CSD can approach machine precision for suitable functions and step sizes
fprime = complex_step_derivative(f, x=1.5, h=1e-20)
# Accuracy depends on f and h; CSD avoids subtractive cancellation.
```

**Why it matters:**
- Has O(h^2) truncation error for suitable analytic functions
- Avoids the usual 1/h amplification of round-off error in finite differences
- Can provide near-machine-precision Jacobians for suitable functions and step sizes
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
| **Iteration details** | ✓ Full transparency | Solver internals are not generally presented for inspection |
| **Stability regions** | ✓ Visualized | ✗ Not exposed |
| **Method education** | ✓ Explicit steps | Production-oriented, not focused on exposing each method step |
| **Jacobian control** | ✓ CSD available | Limited |
| **Speed** | Educational | Optimized |
| **Purpose** | Teaching and experimentation | Production-oriented scientific computing |

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
├── tests/                  # Regression and validation tests
├── README.md               # Project homepage
└── HISTORY.md              # Release history
```

---

## Scope and limitations

NumAnalytica is intentionally focused on transparent numerical methods rather than production-scale solver infrastructure. It is most useful for:

- learning how implicit time-stepping and Newton iteration work in practice
- comparing analytical and numerical differentiation approaches
- inspecting iteration history and convergence behavior in a controlled setting
- reproducing the benchmark experiments and stability analyses used in the thesis

Current limitations include:

- fixed-step solvers rather than a full adaptive-step framework
- no full multistep BDF family beyond the current educational implementation scope
- focus on pedagogical clarity more than industrial optimization
- a smaller test surface than a long-lived production numerical library

This is the appropriate scope for a thesis-ready release: the library demonstrates the core methods and validates the claims of the work without overstating its maturity as a general-purpose scientific software platform.

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

# 5. Export iteration table
table_latex = solver.get_iteration_table()
print(table_latex)
```

---

## Recent fixes and validation

- ✅ `t_eval` is respected by both explicit and implicit Euler solvers
- ✅ `NewtonRaphson(method=...)` follows the selected derivative strategy
- ✅ zero-iteration and failure-case behavior is handled more safely
- ✅ step-size and dimension validation prevents invalid integration inputs from silently propagating
- ✅ RHS/Jacobian consistency checks fail early with clear, actionable error messages
- ✅ regression tests cover the main solver contracts and edge cases
- ✅ appendix benchmark scripts are reproducible via `python verify_examples.py`

## Development Status

- ✅ Core numerical infrastructure
- ✅ Differentiation utilities and Jacobian tools
- ✅ Root-finding methods
- ✅ ODE solvers, including implicit Backward Euler
- ✅ Stability-region analysis and plotting
- ✅ Benchmark and appendix example scripts
- ✅ Reproducible validation workflow for the thesis appendix
- ✅ Thesis-ready release scope and documentation framing

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

**Last Updated:** October 2026
**Status:** Validated pedagogical numerical-analysis library
