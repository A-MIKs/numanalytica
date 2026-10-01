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

        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be positive and finite.")

        h = min(h, max_step)

        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float).ravel()
            if t_eval.size == 0:
                raise ValueError("t_eval must not be empty.")
            if not np.all(np.isfinite(t_eval)):
                raise ValueError("t_eval contains non-finite values.")
            if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
                raise ValueError("t_eval values must lie within [t0, tf].")
            t_points = np.unique(np.concatenate(([t0], t_eval, [tf])))
            t_points = t_points[np.argsort(t_points)]
        else:
            t_points = np.array([t0], dtype=float)
            while t_points[-1] < tf - 1e-15:
                next_t = min(t_points[-1] + h, tf)
                t_points = np.append(t_points, next_t)
                if next_t >= tf - 1e-15:
                    break

        # Build solution array
        t_solution = [t0]
        y_solution = [y0.copy()]

        t = t0
        y = y0.copy()

        iteration = 0

        for target in t_points[1:]:
            if target <= t:
                continue

            # RHS evaluation
            dy = self._evaluate_rhs(t, y, args=args)

            # Euler step
            h_step = min(h, target - t)
            y_next = y + h_step * dy

            # Record iteration
            state = {f"y[{i}]": y[i] for i in range(min(n, 3))}
            self.logger.record_iteration(
                iteration=iteration,
                state={"t": t, **state},
                residual=np.linalg.norm(dy),
                step_length=h_step,
            )

            t = target
            y = y_next
            t_solution.append(t)
            y_solution.append(y.copy())
            iteration += 1

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

        elapsed = time.time() - start_time
        actual_step_sizes = np.diff(np.asarray(t_solution, dtype=float)).tolist()

        result = IntegrationResult(
            solution=y,
            converged=True,
            iterations=iteration,
            residual=0,
            tolerance=0,
            message="integration completed",
            elapsed_time=elapsed,
            t=np.asarray(t_solution),
            y=np.asarray(y_solution),
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._rhs_evals,
            jacobian_evaluations=0,
            step_sizes=actual_step_sizes,
            newton_iterations=[0] * iteration,
        )

        self._print_footer(result)
        return result
