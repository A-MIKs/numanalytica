"""
Standard finite difference formulas for comparison with Complex Step Differentiation.

This module provides classical finite difference approximations for derivatives.
These are included for pedagogical comparison with Complex Step Differentiation.
"""

from typing import Callable, Tuple

import numpy as np


def finite_difference_forward(
    f: Callable,
    x: float,
    h: float = 1e-8,
    args: Tuple = (),
) -> float:
    r"""
    Forward finite difference approximation of the first derivative.

    Formula:
        f'(x) ≈ (f(x+h) - f(x)) / h

    Error: O(h)

    Parameters
    ----------
    f : Callable
        Function to differentiate.
    x : float
        Point of evaluation.
    h : float, default=1e-8
        Step size.
    args : Tuple, optional
        Additional function arguments.

    Returns
    -------
    float
        Approximate derivative value.
    """
    return (f(x + h, *args) - f(x, *args)) / h


def finite_difference_backward(
    f: Callable,
    x: float,
    h: float = 1e-8,
    args: Tuple = (),
) -> float:
    r"""
    Backward finite difference approximation of the first derivative.

    Formula:
        f'(x) ≈ (f(x) - f(x-h)) / h

    Error: O(h)

    Parameters
    ----------
    f : Callable
        Function to differentiate.
    x : float
        Point of evaluation.
    h : float, default=1e-8
        Step size.
    args : Tuple, optional
        Additional function arguments.

    Returns
    -------
    float
        Approximate derivative value.
    """
    return (f(x, *args) - f(x - h, *args)) / h


def finite_difference_centered(
    f: Callable,
    x: float,
    h: float = 1e-8,
    args: Tuple = (),
) -> float:
    r"""
    Centered finite difference approximation of the first derivative.

    Formula:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    Error: O(h²)

    More accurate than forward/backward, but still limited by round-off error.

    Parameters
    ----------
    f : Callable
        Function to differentiate.
    x : float
        Point of evaluation.
    h : float, default=1e-8
        Step size. Critical: h must balance truncation and round-off errors.
    args : Tuple, optional
        Additional function arguments.

    Returns
    -------
    float
        Approximate derivative value.
    """
    return (f(x + h, *args) - f(x - h, *args)) / (2 * h)


def finite_difference_jacobian(
    f: Callable,
    x: np.ndarray,
    method: str = "centered",
    h: float = 1e-8,
    args: Tuple = (),
) -> np.ndarray:
    r"""
    Compute Jacobian matrix using finite differences.

    Parameters
    ----------
    f : Callable
        Vector function F(x) -> R^m.
    x : np.ndarray
        Point of evaluation (shape: (n,)).
    method : str, default="centered"
        One of "forward", "backward", or "centered".
    h : float, default=1e-8
        Step size.
    args : Tuple, optional
        Additional function arguments.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (m, n).

    Notes
    -----
    - Forward/backward: O(h) error
    - Centered: O(h²) error
    - All suffer from round-off when h is too small (~1e-8 is typical)
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # Evaluate at base point
    f_base = f(x, *args)
    f_base = np.atleast_1d(f_base)
    m = f_base.size

    J = np.zeros((m, n))

    if method == "centered":
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += h
            x_minus = x.copy()
            x_minus[j] -= h

            f_plus = np.atleast_1d(f(x_plus, *args))
            f_minus = np.atleast_1d(f(x_minus, *args))

            J[:, j] = (f_plus - f_minus) / (2 * h)

    elif method == "forward":
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += h
            f_plus = np.atleast_1d(f(x_plus, *args))
            J[:, j] = (f_plus - f_base) / h

    elif method == "backward":
        for j in range(n):
            x_minus = x.copy()
            x_minus[j] -= h
            f_minus = np.atleast_1d(f(x_minus, *args))
            J[:, j] = (f_base - f_minus) / h

    else:
        raise ValueError(f"Unknown method: {method}")

    return J


def default_finite_difference_h() -> float:
    """
    Return default step size for finite difference derivatives.

    Uses theoretical optimal h ≈ (6*epsilon)^(1/3) for centered differences,
    where epsilon is machine precision (~1e-16 for float64).

    Returns
    -------
    float
        Default step size (~5.2e-6).
    """
    # Theoretical optimal h ~= (6*eps)^(1/3) ≈ 5.2e-6 for float64
    epsilon = np.finfo(float).eps
    h_opt = (6 * epsilon) ** (1 / 3)
    return h_opt
