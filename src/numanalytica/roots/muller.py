"""Müller's method for root-finding (requires 3 initial points)."""

import time
from typing import Callable, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult


class Muller(BaseSolver):
    """
    Müller's method: quadratic interpolation for root-finding.

    Uses parabolic interpolation through 3 points, can find complex roots
    and converges faster than bisection.

    Parameters
    ----------
    f : callable
        Function to find root of.
    verbose : bool, default=True
        Enable iteration logging.

    Note
    ----
    Requires 3 initial points (x0, x1, x2). Can find complex roots if
    the function accepts complex arguments.
    """

    def __init__(self, f: Callable, verbose: bool = True):
        """Initialize Müller solver."""
        super().__init__(name="Müller's Method", verbose=verbose)
        self.f = f
        self._function_evals = 0

    def solve(
        self,
        x0: float,
        x1: float,
        x2: float,
        tol: float = 1e-9,
        maxiter: int = 100,
        args: Tuple = (),
    ) -> RootResult:
        """
        Find the root using Müller's method.

        Parameters
        ----------
        x0, x1, x2 : float
            Three initial guesses forming a bracket.
        tol : float, default=1e-9
            Convergence tolerance.
        maxiter : int, default=100
            Maximum iterations.
        args : tuple, optional
            Additional function arguments.

        Returns
        -------
        RootResult
            Root and convergence information.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0

        x = np.array([x0, x1, x2], dtype=complex)
        f = np.array([self.f(xi, *args) for xi in x], dtype=complex)
        self._function_evals += 3

        converged = False
        message = "max iterations reached"

        for iteration in range(maxiter):
            # Divided differences
            h0 = x[1] - x[0]
            h1 = x[2] - x[1]

            delta0 = (f[1] - f[0]) / h0
            delta1 = (f[2] - f[1]) / h1

            a = (delta1 - delta0) / (h1 + h0)
            b = delta1 + h1 * a
            c = f[2]

            discriminant = b * b - 4 * a * c

            if abs(a) < 1e-14:
                message = "degenerate case"
                break

            # Two roots; pick closer one
            sqrt_disc = np.sqrt(discriminant)
            denominator1 = b + sqrt_disc
            denominator2 = b - sqrt_disc

            denom_choice = denominator1 if abs(denominator1) > abs(denominator2) else denominator2

            if abs(denom_choice) < 1e-14:
                message = "singularity"
                break

            x_new = x[2] - 2 * c / denom_choice
            f_new = self.f(x_new, *args)
            self._function_evals += 1

            residual = abs(f_new)
            step_length = abs(x_new - x[2])

            self.logger.record_iteration(
                iteration=iteration,
                state={"x": x_new},
                residual=float(residual.real) if isinstance(residual, complex) else float(residual),
                step_length=(
                    float(step_length.real)
                    if isinstance(step_length, complex)
                    else float(step_length)
                ),
            )

            if residual < tol or step_length < tol:
                converged = True
                message = "converged"
                break

            # Shift points
            x[0], x[1], x[2] = x[1], x[2], x_new
            f[0], f[1], f[2] = f[1], f[2], f_new

        elapsed = time.time() - start_time

        # Return real part of complex solution
        x_final = x[2].real if isinstance(x[2], complex) else x[2]

        result = RootResult(
            solution=x_final if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=float(residual.real) if isinstance(residual, complex) else float(residual),
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._function_evals,
        )

        self._print_footer(result)
        return result
