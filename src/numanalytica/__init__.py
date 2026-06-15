"""
NumAnalytica: A pedagogical numerical analysis library for solving ODEs with stability analysis.

This package implements Backward Differentiation Formulas (BDF) for solving stiff
differential equations, with transparent "glass-box" iteration logging and
comprehensive stability analysis.

Key Features:
    - Root-finding methods (Newton-Raphson, Bisection, etc.)
    - Complex Step Differentiation for high-accuracy Jacobians
    - Implicit ODE solvers (Backward Euler, BDF methods)
    - A-Stability analysis and visualization
    - Pedagogical iteration logging for educational transparency
"""

__author__ = "Kola-Ilugbo Ayomikun"
__email__ = "amiks262@gmail.com"
__version__ = "0.2.0"

# Benchmarks module
from numanalytica.benchmarks import van_der_pol, van_der_pol_jacobian

# Core module
from numanalytica.core import (
    BaseSolver,
    ConvergenceError,
    DifferentiationError,
    DivergenceError,
    InitialValueError,
    IntegrationResult,
    IterationLogger,
    NumanalyticalError,
    ODEError,
    RootFindingError,
    RootResult,
    SingularJacobianError,
    SolverResult,
    StabilityAnalysisError,
    StagnationError,
    StepSizeError,
    StiffnessError,
)

# Differentiation module
from numanalytica.differentiation import (
    JacobianComputer,
    complex_step_derivative,
    complex_step_gradient,
    complex_step_jacobian,
    default_finite_difference_h,
    finite_difference_backward,
    finite_difference_centered,
    finite_difference_forward,
    finite_difference_jacobian,
)

# ODE module
from numanalytica.ode import BackwardEuler, BaseIntegrator, ExplicitEuler

# Root-finding module
from numanalytica.roots import (
    Bisection,
    FalsePosition,
    FixedPoint,
    Muller,
    NewtonRaphson,
    NewtonRaphsonSystem,
    Secant,
    get_initial_interval,
)

# Stability module
from numanalytica.stability import (
    StabilityRegion,
    stability_backward_euler,
    stability_bdf2,
    stability_forward_euler,
)

__all__ = [
    # Core
    "BaseSolver",
    "SolverResult",
    "RootResult",
    "IntegrationResult",
    "IterationLogger",
    "NumanalyticalError",
    "RootFindingError",
    "ConvergenceError",
    "StagnationError",
    "DivergenceError",
    "ODEError",
    "StiffnessError",
    "InitialValueError",
    "StepSizeError",
    "DifferentiationError",
    "SingularJacobianError",
    "StabilityAnalysisError",
    # Differentiation
    "complex_step_derivative",
    "complex_step_jacobian",
    "complex_step_gradient",
    "finite_difference_forward",
    "finite_difference_backward",
    "finite_difference_centered",
    "finite_difference_jacobian",
    "JacobianComputer",
    "default_finite_difference_h",
    # Roots
    "NewtonRaphson",
    "NewtonRaphsonSystem",
    "Bisection",
    "Secant",
    "FalsePosition",
    "FixedPoint",
    "Muller",
    "get_initial_interval",
    # ODE
    "BaseIntegrator",
    "ExplicitEuler",
    "BackwardEuler",
    # Stability
    "StabilityRegion",
    "stability_forward_euler",
    "stability_backward_euler",
    "stability_bdf2",
    # Benchmarks
    "van_der_pol",
    "van_der_pol_jacobian",
]
