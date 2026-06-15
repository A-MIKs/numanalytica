"""
Forward (Explicit) Euler method for ODE integration.

This is the simplest explicit ODE solver. It serves as a baseline to
demonstrate why implicit methods (like BDF) are necessary for stiff problems.

Mathematical Formula:
    y_{n+1} = y_n + h * f(t_n, y_n)

Stability: Stability region is |1 + hλ| ≤ 1, severely restricting h for stiff equations.
"""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import IntegrationResult
from numanalytica.ode.base_integrator import BaseIntegrator


class ExplicitEuler(BaseIntegrator):
    """
    Forward Euler method: explicit, first-order ODE integrator.

    WARNING: Use only for non-stiff problems. For stiff equations,
    the step size becomes prohibitively small.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) -> dy/dt.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(self, f: Callable, verbose: bool = True):
        """Initialize Forward Euler solver."""
        super().__init__(f=f, verbose=verbose)
        self.name = "Forward Euler (Explicit)"

    def solve(
        self,
        t0: float,
        tf: float,
        y0: np.ndarray,
        h: Optional[float] = None,
        t_eval: Optional[np.ndarray] = None,
        max_step: float = float("inf"),
        args: Tuple = (),
    ) -> IntegrationResult:
        """
        Integrate using Forward Euler method.

        Parameters
        ----------
        t0, tf : float
            Initial and final times.
        y0 : array_like
            Initial condition.
        h : float, optional
            Fixed step size. If None, adaptive step sizing (not implemented).
        t_eval : ndarray, optional
            Times at which to record solution.
        max_step : float, default=inf
            Maximum allowed step size.
        args : tuple, optional
            Additional arguments to f.

        Returns
        -------
        IntegrationResult
            Solution trajectory.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._rhs_evals = 0

        y0 = self.validate_initial_conditions(t0, tf, y0)
        n = y0.size

        if h is None:
            h = (tf - t0) / 100  # Default: 100 steps

        h = min(h, max_step)

        # Build solution array
        t_solution = []
        y_solution = []

        t = t0
        y = y0.copy()

        iteration = 0

        while t < tf:
            t_solution.append(t)
            y_solution.append(y.copy())

            # RHS evaluation
            dy = self._evaluate_rhs(t, y, args=args)

            # Euler step
            h_step = min(h, tf - t)
            y_next = y + h_step * dy

            # Record iteration
            state = {f"y[{i}]": y[i] for i in range(min(n, 3))}
            self.logger.record_iteration(
                iteration=iteration,
                state={"t": t, **state},
                residual=np.linalg.norm(dy),
                step_length=h_step,
            )

            t += h_step
            y = y_next
            iteration += 1

            # Safety check
            if iteration > 100000:
                return IntegrationResult(
                    solution=None,
                    converged=False,
                    iterations=iteration,
                    residual=np.inf,
                    tolerance=0,
                    message="Too many iterations (step size too small)",
                    elapsed_time=time.time() - start_time,
                    t=np.array(t_solution),
                    y=np.array(y_solution),
                    function_evaluations=self._rhs_evals,
                )

        # Add final point
        t_solution.append(t)
        y_solution.append(y)

        elapsed = time.time() - start_time

        result = IntegrationResult(
            solution=y,
            converged=True,
            iterations=iteration,
            residual=0,  # Explicit method doesn't have convergence criterion
            tolerance=0,
            message="integration completed",
            elapsed_time=elapsed,
            t=np.array(t_solution),
            y=np.array(y_solution),
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._rhs_evals,
            jacobian_evaluations=0,
            step_sizes=[h] * iteration,
            newton_iterations=[0] * iteration,  # No Newton iterations
        )

        self._print_footer(result)
        return result
