"""
Dahlquist test equation utilities.

The Dahlquist equation dy/dt = λy is the standard test for stability analysis.
It allows exact computation of stability regions and convergence rates.

For λ ∈ ℂ, the exact solution is y(t) = y₀ exp(λt).
"""

from typing import Callable

import numpy as np


def dahlquist_rhs(t: float, y: np.ndarray, lam: complex) -> np.ndarray:
    """
    Right-hand side of Dahlquist test equation: dy/dt = λy.

    Parameters
    ----------
    t : float
        Time (not used, autonomous equation).
    y : ndarray
        Current solution.
    lam : complex
        Test parameter λ.

    Returns
    -------
    ndarray
        λy
    """
    return lam * y


def dahlquist_exact(t: float, y0: np.ndarray, lam: complex) -> np.ndarray:
    """
    Exact solution of Dahlquist equation: y(t) = y₀ exp(λt).

    Parameters
    ----------
    t : float
        Time.
    y0 : ndarray
        Initial condition.
    lam : complex
        Test parameter λ.

    Returns
    -------
    ndarray
        Exact solution at time t.
    """
    return y0 * np.exp(lam * t)


def dahlquist_jacobian(t: float, y: np.ndarray, lam: complex) -> np.ndarray:
    """
    Jacobian of Dahlquist equation: ∂(λy)/∂y = λ.

    Parameters
    ----------
    t : float
        Time (not used).
    y : ndarray
        Current solution (not used).
    lam : complex
        Test parameter λ.

    Returns
    -------
    ndarray
        Jacobian matrix (scalar for 1D).
    """
    return np.array([[lam]])


class DahlquistProblem:
    """
    Dahlquist test problem for stability and convergence analysis.

    Parameters
    ----------
    lam : complex
        Test parameter λ (stiffness parameter).
    y0 : ndarray, optional
        Initial condition. Default: [1.0]
    """

    def __init__(self, lam: complex, y0: np.ndarray = None):
        self.lam = lam
        self.y0 = y0 if y0 is not None else np.array([1.0])
        self.rhs = lambda t, y: dahlquist_rhs(t, y, lam)
        self.jacobian = lambda t, y: dahlquist_jacobian(t, y, lam)
        self.exact = lambda t: dahlquist_exact(t, self.y0, lam)

    def __str__(self):
        return f"Dahlquist Problem: dy/dt = {self.lam}y, y(0) = {self.y0}"
