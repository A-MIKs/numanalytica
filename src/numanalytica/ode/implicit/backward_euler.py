"""
Backward Euler method (BDF Order 1): implicit integrator for stiff ODEs.

This is the simplest Backward Differentiation Formula. It is A-stable,
making it suitable for stiff problems where explicit methods fail.

Mathematical Formula (Implicit Equation):
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

This must be solved for y_{n+1} at each time step using Newton-Raphson.

Stability: The stability region covers the entire left half-plane (A-stable).
"""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import IntegrationResult
from numanalytica.ode.base_integrator import BaseIntegrator
from numanalytica.roots import NewtonRaphsonSystem


class BackwardEuler(BaseIntegrator):
    """
    Backward Euler method: implicit, first-order, A-stable ODE integrator.

    Ideal for stiff systems where explicit methods require tiny step sizes.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) -> dy/dt.
    jacobian : callable, optional
        Jacobian ∂f/∂y. If None, uses Complex Step Differentiation.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(
        self,
        f: Callable,
        jacobian: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """Initialize Backward Euler solver."""
        super().__init__(f=f, jacobian=jacobian, verbose=verbose)
        self.name = "Backward Euler (BDF Order 1, A-Stable)"
        self._newton_iters = []

    def solve(
        self,
        t0: float,
        tf: float,
        y0: np.ndarray,
        h: Optional[float] = None,
        t_eval: Optional[np.ndarray] = None,
        newton_tol: float = 1e-6,
        newton_maxiter: int = 10,
        args: Tuple = (),
    ) -> IntegrationResult:
        """
        Integrate using Backward Euler method.

        Parameters
        ----------
        t0, tf : float
            Initial and final times.
        y0 : array_like
            Initial condition.
        h : float, optional
            Fixed step size. Default: (tf - t0) / 100.
        t_eval : ndarray, optional
            Times at which to record solution (not yet used).
        newton_tol : float, default=1e-6
            Tolerance for Newton-Raphson at each step.
        newton_maxiter : int, default=10
            Maximum Newton iterations per step.
        args : tuple, optional
            Additional arguments to f.

        Returns
        -------
        IntegrationResult
            Solution trajectory and convergence statistics.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._rhs_evals = 0
        self._jacobian_evals = 0
        self._newton_iters = []

        y0 = self.validate_initial_conditions(t0, tf, y0)
        n = y0.size

        if h is None:
            h = (tf - t0) / 100

        h = min(h, tf - t0)

        # Solution arrays
        t_solution = [t0]
        y_solution = [y0.copy()]

        t = t0
        y = y0.copy()
        iteration = 0
        last_residual = 0.0

        # Create Newton-Raphson solver once (reuse across steps)
        solver = NewtonRaphsonSystem(
            lambda y_next: y_next,  # Placeholder, will be redefined each step
            jacobian=lambda y_next: np.eye(n),  # Placeholder
            verbose=False,
        )

        while t < tf - h * 0.5:  # Avoid floating-point errors near tf
            h_step = min(h, tf - t)
            t_next = t + h_step

            # Define the implicit equation: G(y_{n+1}) = y_{n+1} - y_n - h*f(t_{n+1}, y_{n+1}) = 0
            def implicit_equation(y_next):
                """The residual function for Newton's method."""
                rhs = self.f(t_next, y_next, *args)
                return y_next - y - h_step * rhs

            def implicit_jacobian(y_next):
                """Jacobian of the implicit equation (for use in Newton's method)."""
                # J = I - h * ∂f/∂y
                J_f = self._evaluate_jacobian(t_next, y_next, args=args)
                return np.eye(n) - h_step * J_f

            # Update solver with current implicit functions
            solver.f = implicit_equation
            solver.jacobian_func = implicit_jacobian
            solver.logger.clear()  # Clear previous step's logs

            # Solve implicit equation using Newton-Raphson
            result_newton = solver.solve(
                x0=y,  # Use previous y as initial guess
                tol=newton_tol,
                maxiter=newton_maxiter,
            )

            if not result_newton.converged:
                return IntegrationResult(
                    solution=None,
                    converged=False,
                    iterations=iteration,
                    residual=result_newton.residual,
                    tolerance=newton_tol,
                    message=f"Newton solver failed at step {iteration}: {result_newton.message}",
                    elapsed_time=time.time() - start_time,
                    t=np.array(t_solution),
                    y=np.array(y_solution),
                    function_evaluations=self._rhs_evals,
                    jacobian_evaluations=self._jacobian_evals,
                )

            y_next = result_newton.root
            self._rhs_evals += result_newton.function_evaluations
            self._jacobian_evals += result_newton.derivative_evaluations
            self._newton_iters.append(result_newton.iterations)

            # Update last residual
            last_residual = result_newton.residual

            # Record iteration
            state = {f"y[{i}]": y_next[i] for i in range(min(n, 3))}
            self.logger.record_iteration(
                iteration=iteration,
                state={"t": t_next, **state},
                residual=result_newton.residual,
                step_length=h_step,
                note=f"Newton iter={result_newton.iterations}",
            )

            t_solution.append(t_next)
            y_solution.append(y_next.copy())

            t = t_next
            y = y_next
            iteration += 1

            # Safety check
            if iteration > 10000:
                return IntegrationResult(
                    solution=None,
                    converged=False,
                    iterations=iteration,
                    residual=np.inf,
                    tolerance=0,
                    message="Too many iterations",
                    elapsed_time=time.time() - start_time,
                    t=np.array(t_solution),
                    y=np.array(y_solution),
                    function_evaluations=self._rhs_evals,
                    jacobian_evaluations=self._jacobian_evals,
                )

        elapsed = time.time() - start_time

        result = IntegrationResult(
            solution=y,
            converged=True,
            iterations=iteration,
            residual=last_residual,
            tolerance=newton_tol,
            message="integration completed successfully",
            elapsed_time=elapsed,
            t=np.array(t_solution),
            y=np.array(y_solution),
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._rhs_evals,
            jacobian_evaluations=self._jacobian_evals,
            step_sizes=[h] * iteration,
            newton_iterations=self._newton_iters,
        )

        self._print_footer(result)
        return result
