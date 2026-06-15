"""
Unified result classes for all solvers.

This module provides standardized output containers for root-finding,
ODE solvers, and other numerical algorithms.

Design Philosophy:
    The SolverResult class encapsulates:
    1. The numerical solution (root, integration trajectory, etc.)
    2. Iteration history (for pedagogical transparency)
    3. Convergence diagnostics (iterations, residuals, etc.)
    4. Performance metrics (computation time, error estimates)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SolverResult:
    """
     Unified result container for all NumAnalytica solvers.

    Attributes:
         solution (float or np.ndarray):
             The numerical solution (root for root-finders, trajectory for ODE solvers).
         converged (bool):
             Whether the solver converged successfully.
         iterations (int):
             Number of iterations performed.
         residual (float):
             Final residual or error measure.
         tolerance (float):
             Tolerance used for convergence criterion.
         message (str):
             Human-readable message explaining the termination condition.
         elapsed_time (float):
             Elapsed computation time in seconds.
         iteration_history (List[Dict[str, Any]]):
             Detailed log of each iteration (for pedagogical transparency).
             Each entry contains: iteration number, current solution, residual, etc.
         diagnostics (Dict[str, Any]):
             Additional solver-specific diagnostic information.
         metadata (Dict[str, Any]):
             Solver metadata (method name, problem size, etc.).
    """

    solution: Any
    converged: bool
    iterations: int
    residual: float
    tolerance: float
    message: str
    elapsed_time: float
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a formatted summary of the result."""
        status = "✓ Converged" if self.converged else "✗ Failed"
        return (
            f"\n{'='*60}\n"
            f"  Solver Result Summary\n"
            f"{'='*60}\n"
            f"Status:            {status}\n"
            f"Iterations:        {self.iterations}\n"
            f"Residual:          {self.residual:.2e}\n"
            f"Tolerance:         {self.tolerance:.2e}\n"
            f"Computation Time:  {self.elapsed_time:.4f} s\n"
            f"Message:           {self.message}\n"
            f"{'='*60}\n"
        )

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"SolverResult(converged={self.converged}, iterations={self.iterations}, "
            f"residual={self.residual:.2e})"
        )

    def get_iteration_table(self) -> str:
        """
        Format iteration history as a human-readable table.

        Returns
        -------
        str
            A formatted table of iteration data.
        """
        if not self.iteration_history:
            return "No iteration history available."

        # Determine column headers from the first entry
        headers = list(self.iteration_history[0].keys())
        col_widths = {h: max(len(h), 12) for h in headers}

        # Adjust widths based on data
        for entry in self.iteration_history:
            for h in headers:
                val_str = str(entry[h])
                col_widths[h] = max(col_widths[h], len(val_str))

        # Build table
        header_line = " | ".join(f"{h:>{col_widths[h]}}" for h in headers)
        separator = "-+-".join("-" * col_widths[h] for h in headers)

        rows = []
        for entry in self.iteration_history:
            row = " | ".join(f"{str(entry[h]):>{col_widths[h]}}" for h in headers)
            rows.append(row)

        return "\n".join([header_line, separator] + rows)


@dataclass
class RootResult(SolverResult):
    """
    Specialized result container for root-finding solvers.

    Additional Attributes
    ----------
    root : float
        Alias for solution (the approximated root).
    bracket : Optional[Tuple[float, float]]
        The final bracketing interval (if applicable).
    function_evaluations : int
        Number of function evaluations.
    derivative_evaluations : int
        Number of derivative evaluations (for Newton-type methods).
    """

    bracket: Optional[Tuple[float, float]] = None
    function_evaluations: int = 0
    derivative_evaluations: int = 0

    @property
    def root(self) -> float:
        """Alias for solution (the approximated root)."""
        return self.solution

    def __str__(self) -> str:
        """Return a formatted summary specific to root-finding."""
        status = "✓ Converged" if self.converged else "✗ Failed"
        evals = f"  Function Evals:    {self.function_evaluations}\n"
        devals = (
            f"  Derivative Evals:  {self.derivative_evaluations}\n"
            if self.derivative_evaluations > 0
            else ""
        )

        # Handle None root for failed solves
        root_str = f"{self.root:.15e}" if self.root is not None else "Not found"

        return (
            f"\n{'='*60}\n"
            f"  Root Finding Result\n"
            f"{'='*60}\n"
            f"Status:            {status}\n"
            f"Root:              {root_str}\n"
            f"Residual:          {self.residual:.2e}\n"
            f"Iterations:        {self.iterations}\n"
            f"{evals}"
            f"{devals}"
            f"Tolerance:         {self.tolerance:.2e}\n"
            f"Computation Time:  {self.elapsed_time:.4f} s\n"
            f"Message:           {self.message}\n"
            f"{'='*60}\n"
        )


@dataclass
class IntegrationResult(SolverResult):
    """
    Specialized result container for ODE integrators.

    Additional Attributes
    ----------
    t : np.ndarray
        Time grid points.
    y : np.ndarray
        Solution trajectory (shape: (n_steps, n_variables)).
    function_evaluations : int
        Number of RHS function evaluations.
    jacobian_evaluations : int
        Number of Jacobian matrix evaluations.
    step_sizes : List[float]
        Individual step sizes used.
    """

    t: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    function_evaluations: int = 0
    jacobian_evaluations: int = 0
    step_sizes: List[float] = field(default_factory=list)
    newton_iterations: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a formatted summary specific to ODE integration."""
        status = "✓ Converged" if self.converged else "✗ Failed"
        return (
            f"\n{'='*60}\n"
            f"  ODE Integration Result\n"
            f"{'='*60}\n"
            f"Status:                {status}\n"
            f"Time span:             [{self.t[0]:.2e}, {self.t[-1]:.2e}]\n"
            f"Solution points:       {len(self.t)}\n"
            f"Variables:             {self.y.shape[1] if len(self.y.shape) > 1 else 1}\n"
            f"RHS Evaluations:       {self.function_evaluations}\n"
            f"Jacobian Evaluations:  {self.jacobian_evaluations}\n"
            f"Avg Newton Iters/Step: {np.mean(self.newton_iterations):.2f}\n"
            f"Computation Time:      {self.elapsed_time:.4f} s\n"
            f"Message:               {self.message}\n"
            f"{'='*60}\n"
        )
