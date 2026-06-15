"""Bisection method for root-finding (bracketing-based)."""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult
from numanalytica.roots.utils import check_bracket_validity, get_initial_interval


class Bisection(BaseSolver):
    """
    Bisection method for root-finding on intervals where f changes sign.

    The bisection method is slow but robust and guaranteed to converge
    for continuous functions.

    Parameters
    ----------
    f : callable
        Function for which to find the root.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(self, f: Callable, verbose: bool = True):
        """Initialize Bisection solver."""
        super().__init__(name="Bisection (Bracketing)", verbose=verbose)
        self.f = f
        self._function_evals = 0

    def solve(
        self,
        bracket: Optional[Tuple[float, float]] = None,
        tol: float = 1e-9,
        maxiter: int = 100,
        args: Tuple = (),
        **kwargs,
    ) -> RootResult:
        """
        Find the root using bisection.

        Parameters
        ----------
        bracket : tuple, optional
            Initial bracketing interval [a, b]. If None, auto-finds one.
        tol : float, default=1e-9
            Convergence tolerance (final bracket width).
        maxiter : int, default=100
            Maximum iterations.
        args : tuple, optional
            Additional function arguments.
        **kwargs
            Passed to get_initial_interval if bracket is None.

        Returns
        -------
        RootResult
            Root and convergence information.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0

        # Get bracket
        if bracket is None:
            try:
                bracket = get_initial_interval(self.f, args=args, **kwargs)
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

        a, b = bracket
        fa = self.f(a, *args)
        fb = self.f(b, *args)
        self._function_evals += 2

        converged = False
        message = "max iterations reached"

        for iteration in range(maxiter):
            c = (a + b) / 2
            fc = self.f(c, *args)
            self._function_evals += 1

            # Width of bracket
            bracket_width = abs(b - a)
            residual = abs(fc)

            self.logger.record_iteration(
                iteration=iteration,
                state={"a": a, "b": b, "c": c},
                residual=residual,
                step_length=bracket_width,
            )

            # Check convergence criteria
            if bracket_width < tol or residual < tol:
                converged = True
                message = "converged on bracket width or residual"
                break

            # Update bracket
            if np.sign(fc) == np.sign(fa):
                a, fa = c, fc
            else:
                b, fb = c, fc

        elapsed = time.time() - start_time

        root = (a + b) / 2
        result = RootResult(
            solution=root if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=abs(self.f(root, *args)) if converged else np.inf,
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            bracket=(a, b),
            function_evaluations=self._function_evals,
        )

        self._print_footer(result)
        return result
