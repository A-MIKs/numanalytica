"""
Core exceptions for NumAnalytica solvers.

This module defines custom exceptions for root-finding, ODE solvers,
and numerical analysis operations.
"""


class NumanalyticalError(Exception):
    """Base exception class for NumAnalytica."""

    pass


class RootFindingError(NumanalyticalError):
    """Base exception for root-finding methods."""

    pass


class ConvergenceError(RootFindingError):
    """Raised when an iterative method fails to converge."""

    pass


class StagnationError(RootFindingError):
    """Raised when successive iterations do not improve the solution."""

    pass


class DivergenceError(RootFindingError):
    """Raised when an iterative method diverges."""

    pass


class BrackettingError(RootFindingError):
    """Raised when a bracketing method cannot find a sign change."""

    pass


class ODEError(NumanalyticalError):
    """Base exception for ODE solvers."""

    pass


class StiffnessError(ODEError):
    """Raised when an explicit method is applied to a stiff problem."""

    pass


class InitialValueError(ODEError):
    """Raised when initial conditions are invalid."""

    pass


class StepSizeError(ODEError):
    """Raised when step size is too large or negative."""

    pass


class DifferentiationError(NumanalyticalError):
    """Raised when numerical differentiation fails."""

    pass


class SingularJacobianError(NumanalyticalError):
    """Raised when the Jacobian matrix is singular."""

    pass


class StabilityAnalysisError(NumanalyticalError):
    """Raised when stability analysis cannot be computed."""

    pass
