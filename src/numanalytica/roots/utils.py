"""
Utilities for root-finding bracketing methods.

This module provides interval and bracketing utilities for root-finding algorithms.
"""

import numpy as np


def get_initial_interval(
    f,
    whole=False,
    positive=True,
    search_range=(-10, 10),
    step=None,
    args=(),
):
    """
    Automatically find an interval [a, b] where a sign change occurs.

    Parameters
    ----------
    f : callable
        The function to analyze.
    whole : bool, default=False
        Restrict search to whole numbers.
    positive : bool, default=True
        Restrict to positive domain.
    search_range : tuple, default=(-10, 10)
        (min, max) range to search.
    step : float, optional
        Step size. Default: 0.1 (or 1 if whole==True).
    args : tuple, optional
        Extra arguments to f.

    Returns
    -------
    list
        Interval [a, b] with opposite-sign endpoints.

    Raises
    ------
    ValueError
        If no sign change found in range.
    """
    if positive:
        search_range = (max(0, search_range[0]), search_range[1])

    a, b = search_range
    step = step or (0.1 if not whole else 1)

    if abs(b - a) < 1:
        step = (b - a) / 10

    x = a
    while x < b:
        x_next = x + step
        if x_next > b:
            x_next = b

        if np.sign(f(x, *args)) != np.sign(f(x_next, *args)):
            return [x, x_next]
        x += step

    raise ValueError(
        f"Could not find sign change in range {search_range}. " f"Try adjusting parameters."
    )


def check_bracket_validity(f, bracket, tol=1e-14, args=()):
    """
    Check if an interval forms a valid bracket (opposite sign endpoints).

    Parameters
    ----------
    f : callable
        The function.
    bracket : tuple or list
        Interval [a, b].
    tol : float, default=1e-14
        Tolerance for zero function value.
    args : tuple, optional
        Extra function arguments.

    Returns
    -------
    bool
        True if bracket is valid (sign change present).
    """
    a, b = bracket
    fa = f(a, *args)
    fb = f(b, *args)

    # Check for sign change
    if np.sign(fa) == np.sign(fb):
        return False

    # Check that neither endpoint is too close to zero
    if np.abs(fa) < tol or np.abs(fb) < tol:
        return True

    return True


def refine_bracket(f, bracket, tol=1e-10, maxiter=50, args=()):
    """
    Refine a bracket by bisection until sufficiently narrow.

    Parameters
    ----------
    f : callable
        The function.
    bracket : tuple
        Initial bracket [a, b].
    tol : float, default=1e-10
        Target bracket width.
    maxiter : int, default=50
        Maximum iterations.
    args : tuple, optional
        Extra arguments to f.

    Returns
    -------
    bracket : tuple
        Refined bracket [a, b].
    """
    a, b = bracket

    for _ in range(maxiter):
        if abs(b - a) < tol:
            break

        c = (a + b) / 2
        if np.sign(f(c, *args)) == np.sign(f(a, *args)):
            a = c
        else:
            b = c

    return (a, b)
