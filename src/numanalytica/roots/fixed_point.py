"""Fixed-point iteration method."""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult


class FixedPoint(BaseSolver):
    """
    Fixed-point iteration: find x such that x = g(x).

    Parameters
    ----------
    g : callable
        Iteration function g(x). Fixed point satisfies x = g(x).
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(self, g: Callable, verbose: bool = True):
        """Initialize Fixed-Point solver."""
        super().__init__(name="Fixed-Point Iteration", verbose=verbose)
        self.g = g
        self._function_evals = 0

    def solve(
        self,
        x0: float,
        tol: float = 1e-9,
        maxiter: int = 100,
        args: Tuple = (),
    ) -> RootResult:
        """
        Find the fixed point x = g(x).

        Parameters
        ----------
        x0 : float
            Initial guess.
        tol : float, default=1e-9
            Convergence tolerance.
        maxiter : int, default=100
            Maximum iterations.
        args : tuple, optional
            Additional function arguments.

        Returns
        -------
        RootResult
            Fixed point and convergence information.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0

        x = float(x0)
        converged = False
        message = "max iterations reached"

        for iteration in range(maxiter):
            x_next = self.g(x, *args)
            self._function_evals += 1

            step_length = abs(x_next - x)

            self.logger.record_iteration(
                iteration=iteration,
                state={"x": x_next},
                residual=step_length,
                step_length=step_length,
            )

            if step_length < tol:
                converged = True
                message = "converged"
                x = x_next
                break

            if np.isnan(x_next) or np.isinf(x_next):
                message = "divergence"
                break

            x = x_next

        elapsed = time.time() - start_time

        result = RootResult(
            solution=x if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=step_length if converged else np.inf,
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._function_evals,
        )

        self._print_footer(result)
        return result
