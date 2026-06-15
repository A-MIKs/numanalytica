"""
Abstract base class for all solvers in NumAnalytica.

This module defines the common interface for all numerical solvers,
ensuring consistency across root-finding methods, ODE integrators, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from numanalytica.core.exceptions import NumanalyticalError
from numanalytica.core.logger import IterationLogger
from numanalytica.core.results import SolverResult


class BaseSolver(ABC):
    """
    Abstract base class for all numerical solvers.

    This class defines the common interface and shared functionality for all
    solvers in NumAnalytica. Subclasses must implement the solve() method.

    Attributes
    ----------
    name : str
        Human-readable name of the solver (e.g., "Newton-Raphson Method").
    verbose : bool
        Controls iteration logging and console output.
    logger : IterationLogger
        Records iteration history for pedagogical transparency.
    """

    def __init__(self, name: str = "BaseSolver", verbose: bool = True):
        """
        Initialize the solver base class.

        Parameters
        ----------
        name : str
            Name of the solver.
        verbose : bool, default=True
            Enable detailed iteration logging.
        """
        self.name = name
        self.verbose = verbose
        self.logger = IterationLogger(verbose=verbose)

    @abstractmethod
    def solve(self, *args, **kwargs) -> SolverResult:
        """
        Solve the problem.

        This method must be implemented by all subclasses. It performs
        the actual numerical computation and returns a SolverResult.

        Returns
        -------
        SolverResult
            Unified result container with solution, convergence info, and history.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{self.name} must implement the solve() method.")

    def _print_header(self) -> None:
        """Print a header for the solver output."""
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"  {self.name}")
            print("=" * 70)

    def _print_footer(self, result: SolverResult) -> None:
        """Print a footer with the result summary."""
        if self.verbose:
            print(result)

    def _validate_inputs(self) -> None:
        """
        Validate solver inputs. Override in subclasses as needed.

        Raises
        ------
        ValueError
            If any input is invalid.
        """
        pass

    def set_verbose(self, verbose: bool) -> None:
        """
        Set verbosity mode.

        Parameters
        ----------
        verbose : bool
            If True, enable iteration logging.
        """
        self.verbose = verbose
        self.logger.verbose = verbose

    def clear_history(self) -> None:
        """Clear iteration history."""
        self.logger.clear()

    def get_iteration_table(self) -> str:
        """
        Get formatted iteration history table.

        Returns
        -------
        str
            Formatted table of iteration details.
        """
        return self.logger.format_table()

    def get_iteration_records(self) -> list:
        """
        Get raw iteration records.

        Returns
        -------
        list
            List of IterationRecord objects.
        """
        return self.logger.get_records()
