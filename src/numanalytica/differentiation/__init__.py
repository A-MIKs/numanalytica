"""Numerical differentiation module: Complex Step, Finite Differences, Jacobians."""

from numanalytica.differentiation.complex_step import (
    complex_step_derivative,
    complex_step_gradient,
    complex_step_jacobian,
    step_size_study,
)
from numanalytica.differentiation.finite_diff import (
    default_finite_difference_h,
    finite_difference_backward,
    finite_difference_centered,
    finite_difference_forward,
    finite_difference_jacobian,
)
from numanalytica.differentiation.jacobian import (
    JacobianComputer,
    check_jacobian_quality,
    compute_jacobian_condition,
    solve_linear_system_with_lu,
)

__all__ = [
    "complex_step_derivative",
    "complex_step_jacobian",
    "complex_step_gradient",
    "step_size_study",
    "finite_difference_forward",
    "finite_difference_backward",
    "finite_difference_centered",
    "finite_difference_jacobian",
    "default_finite_difference_h",
    "JacobianComputer",
    "compute_jacobian_condition",
    "check_jacobian_quality",
    "solve_linear_system_with_lu",
]
