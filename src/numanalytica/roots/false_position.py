"""False Position (Regula Falsi) method for root-finding."""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult
from numanalytica.roots.utils import get_initial_interval


class FalsePosition(BaseSolver):
    """
    False Position (Regula Falsi) method: interpolation-based bracketing.

    Uses linear interpolation between bracket endpoints to estimate root,
    combining speed of secant method with reliability of bisection.

    Parameters
    ----------
    f : callable
        Function to find root of.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(self, f: Callable, verbose: bool = True):
        """Initialize False Position solver."""
        super().__init__(name="False Position (Regula Falsi)", verbose=verbose)
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
        Find the root using false position method.

        Parameters
        ----------
        bracket : tuple, optional
            Initial bracketing interval [a, b]. Auto-finds if None.
        tol : float, default=1e-9
            Convergence tolerance.
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
            # Linear interpolation to find root estimate
            if abs(fb - fa) < 1e-14:
                message = "function values too close"
                break

            c = a - fa * (b - a) / (fb - fa)
            fc = self.f(c, *args)
            self._function_evals += 1

            residual = abs(fc)
            bracket_width = abs(b - a)

            self.logger.record_iteration(
                iteration=iteration,
                state={"a": a, "b": b, "c": c},
                residual=residual,
                step_length=bracket_width,
            )

            if bracket_width < tol or residual < tol:
                converged = True
                message = "converged"
                break

            # Update bracket
            if np.sign(fc) == np.sign(fa):
                a, fa = c, fc
            else:
                b, fb = c, fc

        elapsed = time.time() - start_time

        root = c if converged else (a + b) / 2
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
