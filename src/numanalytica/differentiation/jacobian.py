"""
Jacobian matrix computation utilities.

This module provides utilities to compute and manage Jacobian matrices
using different numerical differentiation methods (CSD, finite differences, etc.).
"""

from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from numanalytica.differentiation.complex_step import complex_step_jacobian
from numanalytica.differentiation.finite_diff import finite_difference_jacobian


class JacobianComputer:
    """
    Manages Jacobian computation with various methods.

    This class provides a unified interface for computing Jacobians
    using Complex Step Differentiation, finite differences, or user-provided
    analytical Jacobians.

    Attributes
    ----------
    method : str
        Method used: "complex_step", "finite_diff", or "analytical".
    order : int
        Number of times the Jacobian has been evaluated.
    n_evals : int
        Total count of Jacobian evaluations.
    """

    def __init__(
        self,
        f: Callable,
        method: Literal["complex_step", "finite_diff", "analytical"] = "complex_step",
        jacobian_func: Optional[Callable] = None,
        h: float = 1e-20,
        fd_method: str = "centered",
    ):
        """
        Initialize Jacobian computer.

        Parameters
        ----------
        f : Callable
            Vector function F: R^n -> R^m.
        method : str, default="complex_step"
            Differentiation method. Options:
            - "complex_step": Use Complex Step Differentiation(most accurate)
            - "finite_diff": Use finite differences
            - "analytical": Use provided jacobian_func
        jacobian_func : Callable, optional
            User-provided Jacobian function. Required if method="analytical".
        h : float, default=1e-20
            Step size for complex step or finite differences.
        fd_method : str, default="centered"
            Finite difference method: "forward", "backward", or "centered".
        """
        self.f = f
        self.method = method
        self.jacobian_func = jacobian_func
        self.h = h
        self.fd_method = fd_method
        self.n_evals = 0

        if method == "analytical" and jacobian_func is None:
            raise ValueError("jacobian_func must be provided when method='analytical'")

    def __call__(self, x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Compute the Jacobian at a given point.

        Parameters
        ----------
        x : np.ndarray
            Point of evaluation (shape: (n,)).
        *args, **kwargs
            Additional arguments for the function.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix of shape (m, n).
        info : dict
            Metadata: {"method": str, "evals": int, "h": float, ...}
        """
        self.n_evals += 1

        if self.method == "analytical":
            J = self.jacobian_func(x, *args, **kwargs)
            info = {
                "method": "analytical",
                "evals": 1,
                "h": None,
            }
        elif self.method == "complex_step":
            J = complex_step_jacobian(self.f, x, h=self.h)
            info = {
                "method": "complex_step",
                "evals": x.size,  # One function eval per column
                "h": self.h,
            }
        elif self.method == "finite_diff":
            J = finite_difference_jacobian(self.f, x, method=self.fd_method, h=self.h)
            evals_per_col = 2 if self.fd_method == "centered" else 1
            info = {
                "method": "finite_diff",
                "fd_method": self.fd_method,
                "evals": evals_per_col * x.size,
                "h": self.h,
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")

        info["total_evals"] = self.n_evals

        return J, info

    def reset(self) -> None:
        """Reset evaluation counter."""
        self.n_evals = 0

    def summary(self) -> str:
        """Return a summary of Jacobian computations."""
        return f"JacobianComputer(method={self.method}, " f"total_evals={self.n_evals})"


def solve_linear_system_with_lu(
    J: np.ndarray,
    b: np.ndarray,
    use_lu_cache: bool = False,
    lu_cache: Optional[Tuple] = None,
) -> Tuple[np.ndarray, Optional[Tuple], dict]:
    """
    Solve a linear system J*x = b using LU factorization.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix (m, n) where m >= n.
    b : np.ndarray
        Right-hand side vector (m,).
    use_lu_cache : bool, default=False
        If True, reuse cached LU decomposition.
    lu_cache : Tuple, optional
        Cached LU decomposition from scipy.linalg.lu_factor.

    Returns
    -------
    x : np.ndarray
        Solution vector.
    lu_cache : Tuple or None
        LU decomposition (if computed), for reuse in next iteration.
    info : dict
        Solver information.
    """
    from scipy.linalg import lu_factor, lu_solve

    try:
        if use_lu_cache and lu_cache is not None:
            # Reuse cached factorization
            x = lu_solve(lu_cache, b)
            info = {"method": "lu_cached", "refactored": False}
        else:
            # Compute new factorization
            lu, piv = lu_factor(J)
            x = lu_solve((lu, piv), b)
            lu_cache = (lu, piv)
            info = {"method": "lu_solved", "refactored": True}

        info["condition_number"] = np.linalg.cond(J)
        info["rank"] = np.linalg.matrix_rank(J)
        info["singular"] = False

    except np.linalg.LinAlgError:
        x = np.full_like(b, np.nan)
        info = {
            "method": "lu_failed",
            "singular": True,
            "error": "Singular Jacobian matrix",
        }

    return x, lu_cache, info


def compute_jacobian_condition(J: np.ndarray) -> float:
    """
    Compute condition number of the Jacobian.

    A large condition number indicates that the Jacobian is nearly singular,
    which can cause Newton-Raphson convergence problems.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix.

    Returns
    -------
    float
        Condition number (2-norm).
    """
    return np.linalg.cond(J, p=2)


def check_jacobian_quality(
    J: np.ndarray,
    condition_threshold: float = 1e10,
) -> dict:
    """
    Assess the quality and conditioning of a Jacobian matrix.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix.
    condition_threshold : float
        Threshold above which we flag ill-conditioning.

    Returns
    -------
    dict
        Quality assessment with keys:
        - "condition_number": float
        - "is_singular": bool
        - "is_ill_conditioned": bool
        - "rank": int
        - "rank_deficient": bool
        - "warnings": list of str
    """
    warnings = []

    try:
        cond = compute_jacobian_condition(J)
    except Exception as e:
        return {
            "condition_number": np.inf,
            "is_singular": True,
            "is_ill_conditioned": True,
            "rank": 0,
            "rank_deficient": True,
            "warnings": [f"Error computing condition: {e}"],
        }

    rank = np.linalg.matrix_rank(J)
    m, n = J.shape
    is_singular = rank < min(m, n)
    is_ill_conditioned = cond > condition_threshold

    if is_singular:
        warnings.append(f"Jacobian is singular (rank={rank}, expected >={min(m,n)})")
    if is_ill_conditioned:
        warnings.append(f"Jacobian is ill-conditioned (κ={cond:.2e})")

    return {
        "condition_number": cond,
        "is_singular": is_singular,
        "is_ill_conditioned": is_ill_conditioned,
        "rank": rank,
        "rank_deficient": rank < min(m, n),
        "warnings": warnings,
    }
