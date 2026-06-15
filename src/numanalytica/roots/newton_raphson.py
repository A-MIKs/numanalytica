"""
Newton-Raphson root-finding method with Complex Step Differentiation support.

This is a key solver for the implicit ODE integrators, as it solves the
nonlinear equations that arise at each time step of BDF methods.
"""

import time
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, RootResult
from numanalytica.core.exceptions import ConvergenceError, DivergenceError
from numanalytica.differentiation import JacobianComputer, complex_step_derivative
from numanalytica.roots.utils import get_initial_interval


class NewtonRaphson(BaseSolver):
    """
    Newton-Raphson method for finding roots with high-accuracy derivatives.

    This solver uses Complex Step Differentiation by default to compute
    derivatives with machine precision, avoiding round-off error.

    Mathematical Background:
        The Newton-Raphson method iterates:
            x_{n+1} = x_n - f(x_n) / f'(x_n)

        For systems, replaces with:
            x_{n+1} = x_n - J^{-1}(x_n) * f(x_n)

        where J is the Jacobian matrix.

    Parameters
    ----------
    f : callable
        Function or system F(x) -> float or array.
    fprime : callable, optional
        Analytical derivative f'(x). If None, uses Complex Step Differentiation.
    method : str, default="complex_step"
        Derivative method: "complex_step", "finite_diff", or "analytical".
    verbose : bool, default=True
        Enable iteration logging.

    Examples
    --------
    >>> def f(x): return x**2 - 4
    >>> solver = NewtonRaphson(f)
    >>> result = solver.solve(x0=1.0)
    >>> print(result.root)  # ≈ 2.0
    """

    def __init__(
        self,
        f: Callable,
        fprime: Optional[Callable] = None,
        method: str = "complex_step",
        verbose: bool = True,
    ):
        """Initialize Newton-Raphson solver."""
        super().__init__(name="Newton-Raphson", verbose=verbose)
        self.f = f
        self.fprime = fprime
        self.method = method
        self._function_evals = 0
        self._derivative_evals = 0

    def solve(
        self,
        x0: Optional[float] = None,
        tol: float = 1e-9,
        maxiter: int = 100,
        bracket: Optional[Tuple[float, float]] = None,
        args: Tuple = (),
        **kwargs,
    ) -> RootResult:
        """
        Solve for the root.

        Parameters
        ----------
        x0 : float, optional
            Initial guess. If None, attempts automatic bracketing.
        tol : float, default=1e-9
            Convergence tolerance.
        maxiter : int, default=100
            Maximum iterations.
        bracket : tuple, optional
            Bracketing interval (alternative to x0).
        args : tuple, optional
            Additional arguments to f and fprime.
        **kwargs
            Passed to get_initial_interval if x0 is None.

        Returns
        -------
        RootResult
            Structured result with root, convergence info, iteration history.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0
        self._derivative_evals = 0

        # Initial guess
        if x0 is None and bracket is None:
            try:
                interval = get_initial_interval(self.f, args=args, **kwargs)
                x0 = sum(interval) / 2
            except ValueError as e:
                return RootResult(
                    solution=None,
                    converged=False,
                    iterations=0,
                    residual=np.inf,
                    tolerance=tol,
                    message=f"Failed to find initial bracket: {e}",
                    elapsed_time=time.time() - start_time,
                )

        x = float(x0)
        converged = False
        message = "max iterations reached"

        # Setup derivative computation
        if self.fprime is None:
            # Use automatic differentiation
            def compute_derivative(x_val):
                self._derivative_evals += 1
                return complex_step_derivative(self.f, x_val, args=args)

        else:
            compute_derivative = lambda x_val: self.fprime(x_val, *args)

        # Main iteration loop
        for iteration in range(maxiter):
            f_x = self.f(x, *args)
            self._function_evals += 1
            residual = abs(f_x)

            # Record iteration
            self.logger.record_iteration(
                iteration=iteration,
                state={"x": x},
                residual=residual,
                function_evals=self._function_evals,
                derivative_evals=self._derivative_evals,
                step_length=None,
            )

            # Check for convergence
            if residual < tol:
                converged = True
                message = "converged on residual"
                break

            # Compute derivative
            fprime_x = compute_derivative(x)

            # Check for zero derivative
            if abs(fprime_x) < 1e-14:
                message = "zero derivative"
                break

            # Newton step
            step = f_x / fprime_x
            x_next = x - step

            # Check for divergence
            if np.isnan(x_next) or np.isinf(x_next):
                message = "divergence detected"
                break

            # Check for convergence on step size
            if abs(x_next - x) < tol:
                x = x_next
                f_x = self.f(x, *args)
                self._function_evals += 1
                converged = True
                message = "converged on step size"
                break

            x = x_next

        elapsed = time.time() - start_time

        result = RootResult(
            solution=x if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=abs(f_x) if converged else np.inf,
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._function_evals,
            derivative_evaluations=self._derivative_evals,
        )

        self._print_footer(result)
        return result


class NewtonRaphsonSystem(BaseSolver):
    """
    Newton-Raphson for systems of nonlinear equations F(x) = 0.

    For a system F: R^n -> R^n, solves:
        x_{n+1} = x_n - J(x_n)^{-1} * F(x_n)

    where J is the Jacobian matrix.

    Parameters
    ----------
    f : callable
        System function F(x) -> np.ndarray of shape (n,).
    jacobian : callable, optional
        Analytical Jacobian J(x). If None, uses Complex Step Differentiation.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(
        self,
        f: Callable,
        jacobian: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """Initialize Newton-Raphson system solver."""
        super().__init__(name="Newton-Raphson (System)", verbose=verbose)
        self.f = f
        self.jacobian_func = jacobian
        self._function_evals = 0
        self._jacobian_evals = 0

    def solve(
        self,
        x0: np.ndarray,
        tol: float = 1e-9,
        maxiter: int = 100,
        use_lu_cache: bool = True,
        args: Tuple = (),
    ) -> RootResult:
        """
        Solve the system F(x) = 0.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess (shape: (n,)).
        tol : float, default=1e-9
            Convergence tolerance.
        maxiter : int, default=100
            Maximum iterations.
        use_lu_cache : bool, default=True
            Cache LU decomposition across iterations.
        args : tuple, optional
            Additional function arguments.

        Returns
        -------
        RootResult
            Solution and convergence diagnostics.
        """
        self._print_header()
        start_time = time.time()
        self.logger.clear()
        self._function_evals = 0
        self._jacobian_evals = 0

        from numanalytica.differentiation import (
            JacobianComputer,
            solve_linear_system_with_lu,
        )

        x = np.asarray(x0, dtype=float)
        converged = False
        message = "max iterations reached"
        lu_cache = None

        # Setup Jacobian computation
        jacobian_computer = JacobianComputer(
            self.f,
            method="analytical" if self.jacobian_func else "complex_step",
            jacobian_func=self.jacobian_func,
        )

        for iteration in range(maxiter):
            # Evaluate function
            F_x = np.asarray(self.f(x, *args), dtype=float).flatten()
            self._function_evals += 1

            # Compute residual (norm of F)
            residual = np.linalg.norm(F_x)

            # Record iteration
            state = {f"x[{i}]": x[i] for i in range(len(x))}
            self.logger.record_iteration(
                iteration=iteration,
                state=state,
                residual=residual,
                function_evals=self._function_evals,
                derivative_evals=self._jacobian_evals,
            )

            # Check convergence
            if residual < tol:
                converged = True
                message = "converged"
                break

            # Compute Jacobian
            J, jac_info = jacobian_computer(x, *args)
            self._jacobian_evals += jac_info["evals"]

            # Solve linear system J*step = -F
            step, lu_cache, solve_info = solve_linear_system_with_lu(
                J, -F_x, use_lu_cache=use_lu_cache, lu_cache=lu_cache
            )

            if solve_info.get("singular", False):
                message = "singular jacobian"
                break

            # Update solution
            x_next = x + step
            step_norm = np.linalg.norm(step)

            # Check for convergence on step size
            if step_norm < tol:
                x = x_next
                converged = True
                message = "converged on step size"
                break

            # Check for divergence
            if np.any(np.isnan(x_next)) or np.any(np.isinf(x_next)):
                message = "divergence"
                break

            x = x_next

        elapsed = time.time() - start_time

        result = RootResult(
            solution=x if converged else None,
            converged=converged,
            iterations=iteration + 1,
            residual=residual,
            tolerance=tol,
            message=message,
            elapsed_time=elapsed,
            iteration_history=self.logger.to_dict_list(),
            function_evaluations=self._function_evals,
            derivative_evaluations=self._jacobian_evals,
        )

        self._print_footer(result)
        return result
