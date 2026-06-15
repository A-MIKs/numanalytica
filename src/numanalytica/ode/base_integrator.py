"""
Base class for Initial Value Problem (IVP) solvers.

This module defines the common interface for ODE integrators,
both explicit and implicit methods.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np

from numanalytica.core import BaseSolver, IntegrationResult


class BaseIntegrator(BaseSolver):
    """
    Abstract base class for Initial Value Problem (IVP) solvers.

    Solves the first-order ODE system:
        dy/dt = f(t, y),  y(t0) = y0

    on the interval [t0, tf].

    Parameters
    ----------
    f : callable
        Right-hand side function f(t, y) -> dy/dt.
        Signature: f(t, y) where y can be scalar or array.
    jacobian : callable, optional
        Jacobian matrix J(t, y) = ∂f/∂y, needed for implicit methods.
    dense_out : bool, default=False
        If True, can produce solution at arbitrary intermediate times.
    verbose : bool, default=True
        Enable iteration logging.
    """

    def __init__(
        self,
        f: Callable,
        jacobian: Optional[Callable] = None,
        dense_out: bool = False,
        verbose: bool = True,
    ):
        """Initialize the ODE integrator."""
        super().__init__(name="BaseIntegrator", verbose=verbose)
        self.f = f
        self.jacobian = jacobian
        self.dense_out = dense_out
        self._function_evals = 0
        self._jacobian_evals = 0
        self._rhs_evals = 0

    def validate_initial_conditions(
        self,
        t0: float,
        tf: float,
        y0: np.ndarray,
    ) -> np.ndarray:
        """
        Validate initial conditions and parameters.

        Parameters
        ----------
        t0, tf : float
            Initial and final times.
        y0 : np.ndarray
            Initial condition vector.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        from numanalytica.core.exceptions import InitialValueError, StepSizeError

        if not np.isfinite(t0) or not np.isfinite(tf):
            raise InitialValueError(f"Invalid time bounds: t0={t0}, tf={tf}")

        if t0 >= tf:
            raise StepSizeError(f"Must have t0 < tf, got t0={t0}, tf={tf}")

        y0 = np.asarray(y0, dtype=float).flatten()
        if not np.all(np.isfinite(y0)):
            raise InitialValueError(f"Initial condition contains non-finite values")

        return y0

    @abstractmethod
    def solve(
        self,
        t0: float,
        tf: float,
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        **kwargs,
    ) -> IntegrationResult:
        """
        Integrate the ODE from t0 to tf.

        Parameters
        ----------
        t0 : float
            Initial time.
        tf : float
            Final time (or a time at which to stop).
        y0 : array_like
            Initial condition y(t0).
        t_eval : ndarray, optional
            Times at which to evaluate the solution.
            If None, solver chooses adaptive steps.
        **kwargs
            Method-specific options.

        Returns
        -------
        IntegrationResult
            Solution trajectory and convergence information.
        """
        raise NotImplementedError(f"{self.name} must implement solve()")

    def _evaluate_rhs(
        self,
        t: float,
        y: np.ndarray,
        args: Tuple = (),
    ) -> np.ndarray:
        """
        Evaluate the right-hand side f(t, y) with counting.

        Parameters
        ----------
        t : float
            Time.
        y : np.ndarray
            Current solution.
        args : tuple, optional
            Additional arguments to f.

        Returns
        -------
        np.ndarray
            dy/dt = f(t, y).
        """
        self._rhs_evals += 1
        return np.asarray(self.f(t, y, *args), dtype=float).flatten()

    def _evaluate_jacobian(
        self,
        t: float,
        y: np.ndarray,
        args: Tuple = (),
    ) -> np.ndarray:
        """
        Evaluate the Jacobian ∂f/∂y with counting.

        Parameters
        ----------
        t : float
            Time.
        y : np.ndarray
            Current solution.
        args : tuple, optional
            Additional arguments.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (n, n).
        """
        if self.jacobian is None:
            # Use automatic differentiation if not provided
            from numanalytica.differentiation import complex_step_jacobian

            def f_system(y_var):
                return self.f(t, y_var, *args)

            J = complex_step_jacobian(f_system, y)
        else:
            J = np.asarray(self.jacobian(t, y), dtype=float)

        self._jacobian_evals += 1
        return J

    def get_statistics(self) -> dict:
        """
        Return statistics on solver performance.

        Returns
        -------
        dict
            Statistics dictionary.
        """
        return {
            "rhs_evaluations": self._rhs_evals,
            "jacobian_evaluations": self._jacobian_evals,
        }
