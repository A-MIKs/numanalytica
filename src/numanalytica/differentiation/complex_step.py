"""
Complex Step Differentiation for high-accuracy numerical derivatives.

This module implements Complex Step Differentiation (CSD), a powerful numerical
differentiation technique that avoids subtractive cancellation error by using
complex-valued arithmetic.

Mathematical Foundation:
    For a real-valued function f(x), the derivative can be approximated as:

        f'(x) ≈ Im(f(x + ih)) / h

    where h is a small real step. For suitable functions and step sizes, CSD
    can achieve near machine precision by avoiding subtractive cancellation.
    The function must preserve the complex perturbation and have suitable
    analytic behavior near the evaluation point.

Key Advantages over Finite Differences:
    1. The step size is not subject to the same truncation/round-off balance
       as finite differences.
    2. The derivative estimate has O(h²) truncation error for suitable smooth,
       analytic functions, without the usual 1/h amplification of round-off
       in finite-difference formulas. Floating-point and implementation errors
       still remain.
    3. Useful for Newton-Raphson iterations in implicit solvers.

References:
    Martins, J.R.R.A., Sturdza, P., & Alonso, J.J. (2003).
    "The complex-step derivative approximation."
    ACM Transactions on Mathematical Software, 29(3), 245-262.
"""

from typing import Callable, Optional, Tuple

import numpy as np


def complex_step_derivative(
    f: Callable,
    x: float,
    h: float = 1e-20,
    args: Tuple = (),
) -> float:
    r"""
    Compute first derivative using Complex Step Differentiation.

    Parameters
    ----------
    f : Callable
        Function for which to compute the derivative.
        Must accept complex-valued inputs and preserve the imaginary
        perturbation in its output. CSD is generally suitable for functions
        with an analytic extension near x; real casts or operations that
        discard or alter the imaginary component can give incorrect results.
        Signature: f(x, *args) -> float or complex
    x : float
        Point at which to evaluate the derivative.
    h : float, default=1e-20
        Step size. CSD avoids the usual subtractive cancellation and 1/h
        round-off amplification of finite differences, but accuracy still
        depends on the function and step size. Default is 1e-20.
    args : Tuple, optional
        Additional arguments to pass to f.

    Returns
    -------
    float
        Estimated value of f'(x) with high accuracy.

    Notes
    -----
    For suitable functions, the O(h²) truncation error can be made small with
    a very small h without the usual 1/h round-off amplification found in
    finite differences. This does not eliminate floating-point error, and
    the function must preserve the complex perturbation and behave analytically
    near x. A function that casts inputs to real values or discards the
    imaginary component may return an incorrect derivative.

    Examples
    --------
    >>> def f(x): return x**3 - 2*x
    >>> deriv = complex_step_derivative(f, 2.0)
    >>> print(f"{deriv:.15f}")  # Should be exactly 10.0
    10.000000000000000
    """
    # Evaluate f at x + ih
    f_complex = f(x + 1j * h, *args)

    # Extract imaginary part and divide by h
    derivative = np.imag(f_complex) / h

    return float(derivative)


def complex_step_jacobian(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-20,
    args: Tuple = (),
) -> np.ndarray:
    r"""
    Compute Jacobian matrix using Complex Step Differentiation.

    For a system of equations F: R^n -> R^m, computes the m×n Jacobian
    matrix where J[i,j] = ∂F_i / ∂x_j.

    Parameters
    ----------
    f : Callable
        Vector function F(x) returning an array.
        Signature: f(x, *args) -> np.ndarray of shape (m,)
    x : np.ndarray
        Point at which to evaluate the Jacobian (shape: (n,)).
    h : float, default=1e-20
        Complex step size.
    args : Tuple, optional
        Additional arguments to pass to f.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (m, n).

    Notes
    -----
    The function must preserve complex inputs and have an analytic extension
    near x. Merely accepting complex-valued inputs is not sufficient: real
    casts, non-analytic operations, or other operations that discard or alter
    the imaginary perturbation can produce incorrect derivatives. Accuracy
    depends on the function and step size; CSD avoids the usual subtractive
    cancellation and 1/h round-off amplification of finite differences, but
    does not eliminate all floating-point error.

    Examples
    --------
    >>> def F(x):
    ...     return np.array([x[0]**2 + x[1], x[0]*x[1] - 1])
    >>> x = np.array([1.0, 2.0])
    >>> J = complex_step_jacobian(F, x)
    >>> print(J)  # Should be [[2., 1.], [2., 1.]]
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # Evaluate at base point to get output dimension
    f_base = f(x, *args)
    f_base = np.atleast_1d(f_base)
    m = f_base.size

    # Initialize Jacobian
    J = np.zeros((m, n))

    # Compute each column of the Jacobian
    for j in range(n):
        x_perturb = x.copy().astype(complex)  # Ensure complex dtype
        x_perturb[j] += 1j * h

        f_perturb = f(x_perturb, *args)
        f_perturb = np.atleast_1d(f_perturb)

        # Extract imaginary part and divide by h
        J[:, j] = np.imag(f_perturb) / h

    return J


def complex_step_gradient(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-20,
    args: Tuple = (),
) -> np.ndarray:
    r"""
    Compute gradient vector using Complex Step Differentiation.

    For a scalar function f: R^n -> R, computes the gradient ∇f.

    Parameters
    ----------
    f : Callable
        Scalar function mapping R^n -> R.
        Signature: f(x, *args) -> float
    x : np.ndarray
        Point at which to evaluate the gradient (shape: (n,)).
    h : float, default=1e-20
        Complex step size.
    args : Tuple, optional
        Additional arguments to pass to f.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (n,).

    Examples
    --------
    >>> def f(x): return x[0]**2 + 2*x[1]**2
    >>> x = np.array([1.0, 2.0])
    >>> grad = complex_step_gradient(f, x)
    >>> print(grad)  # Should be [2., 8.]
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    grad = np.zeros(n)

    for i in range(n):
        x_perturb = x.copy().astype(complex)  # Ensure complex dtype
        x_perturb[i] += 1j * h
        f_perturb = f(x_perturb, *args)
        grad[i] = np.imag(f_perturb) / h

    return grad


def step_size_study(
    f: Callable,
    x: float,
    fprime_exact: Optional[Callable] = None,
    h_values: Optional[np.ndarray] = None,
    args: Tuple = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Study derivative accuracy as a function of step size.

    This function evaluates the CSD approximation for multiple step sizes,
    useful for understanding numerical behavior and selecting optimal h.

    Parameters
    ----------
    f : Callable
        Function for which to compute derivatives.
    x : float
        Point of evaluation.
    fprime_exact : Callable, optional
        Exact derivative function for error calculation.
        If provided, absolute errors are computed.
    h_values : np.ndarray, optional
        Array of step sizes to test. Defaults to [1e-20, 1e-15, ..., 1e-5].
    args : Tuple, optional
        Additional arguments to f.

    Returns
    -------
    h_values : np.ndarray
        Step sizes tested.
    derivatives : np.ndarray
        Derivative estimates at each step size.
    errors : np.ndarray
        Absolute errors if fprime_exact is provided, otherwise None.
    """
    if h_values is None:
        h_values = np.logspace(-20, -5, 50)

    derivatives = np.zeros_like(h_values)
    errors = np.zeros_like(h_values) if fprime_exact is not None else None

    fprime_exact_val = fprime_exact(x) if fprime_exact is not None else None

    for i, h in enumerate(h_values):
        deriv = complex_step_derivative(f, x, h=h, args=args)
        derivatives[i] = deriv
        if fprime_exact is not None:
            errors[i] = np.abs(deriv - fprime_exact_val)

    return h_values, derivatives, errors
