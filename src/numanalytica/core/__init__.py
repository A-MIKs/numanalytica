"""Core module: Foundation classes, logging, and exceptions for NumAnalytica."""

from numanalytica.core.base_solver import BaseSolver
from numanalytica.core.exceptions import (
    BrackettingError,
    ConvergenceError,
    DifferentiationError,
    DivergenceError,
    InitialValueError,
    NumanalyticalError,
    ODEError,
    RootFindingError,
    SingularJacobianError,
    StabilityAnalysisError,
    StagnationError,
    StepSizeError,
    StiffnessError,
)
from numanalytica.core.logger import IterationLogger, IterationRecord
from numanalytica.core.results import IntegrationResult, RootResult, SolverResult

__all__ = [
    "BaseSolver",
    "IterationLogger",
    "IterationRecord",
    "SolverResult",
    "RootResult",
    "IntegrationResult",
    "NumanalyticalError",
    "RootFindingError",
    "ConvergenceError",
    "StagnationError",
    "DivergenceError",
    "BrackettingError",
    "ODEError",
    "StiffnessError",
    "InitialValueError",
    "StepSizeError",
    "DifferentiationError",
    "SingularJacobianError",
    "StabilityAnalysisError",
]
