"""Secant method - derivative-free variant of Newton-Raphson."""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult
from numanalytica.roots.utils import get_initial_interval


class Secant(BaseSolver):
    """
    Secant method: derivative-free variant of Newton-Raphson.

    Uses two previous approximations to estimate the derivative,
    avoiding explicit derivative computation.

    Parameters
    ----------
    f : callable
        Function to find root of.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(self, f: Callable, verbose: bool = True):
        """Initialize Secant solver."""
        super().__init__(name="Secant Method", verbose=verbose)
        self.f = f
        self._function_evals = 0

    def solve(
        self,
        x0: Optional[float] = None,
        x1: Optional[float] = None,
        tol: float = 1e-9,
        maxiter: int = 100,
        args: Tuple = (),
        **kwargs,
    ) -> RootResult:
        """
        Find the root using the secant method.

        Parameters
        ----------
        x0, x1 : float, optional
            Two initial guesses. If not provided, auto-bracketing is used.
        tol : float, default=1e-9
            Convergence tolerance.
        maxiter : int, default=100
            Maximum iterations.
        args : tuple, optional
            Additional function arguments.
        **kwargs
            Passed to get_initial_interval if x0 is None.

        Returns
        -------
        RootResult
            Root and convergence information.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0

        # Get initial guesses
        if x0 is None:
            try:
                interval = get_initial_interval(self.f, args=args, **kwargs)
                x0, x1 = interval
            except ValueError as e:
                return RootResult(
                    solution=None,
                    converged=False,
                    iterations=0,
                    residual=np.inf,
                    tolerance=tol,
                    message=f"Cannot find initial bracket: {e}",
                    elapsed_time=time.time() - start_time,
                )

        f0 = self.f(x0, *args)
        f1 = self.f(x1, *args)
        self._function_evals += 2

        converged = False
        message = "max iterations reached"

        for iteration in range(maxiter):
            denominator = f1 - f0
            if abs(denominator) < 1e-14:
                message = "function values too close"
                break

            # Secant step
            x2 = x1 - f1 * (x1 - x0) / denominator
            f2 = self.f(x2, *args)
            self._function_evals += 1

            residual = abs(f2)
            step_length = abs(x2 - x1)

            self.logger.record_iteration(
                iteration=iteration,
                state={"x": x2},
                residual=residual,
                step_length=step_length,
            )

            if residual < tol or step_length < tol:
                converged = True
                message = "converged"
                break

            x0, f0 = x1, f1
            x1, f1 = x2, f2

        elapsed = time.time() - start_time

        result = RootResult(
            solution=x1 if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=abs(f1) if converged else np.inf,
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._function_evals,
        )

        self._print_footer(result)
        return result
