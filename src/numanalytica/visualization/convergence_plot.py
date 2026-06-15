"""
Convergence plotting utilities for numerical methods.

Provides functions to visualize convergence behavior, error vs step size,
and iteration history for root-finding and ODE integration methods.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_history(
    iterations: List[dict],
    title: str = "Convergence History",
    residual_key: str = "residual",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot convergence history from iteration data.

    Parameters
    ----------
    iterations : list of dict
        Iteration history with residual information.
    title : str
        Plot title.
    residual_key : str, default="residual"
        Key for residual values in iteration dicts.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    residuals = [iter.get(residual_key, np.inf) for iter in iterations]
    iterations_nums = list(range(len(residuals)))

    ax.semilogy(iterations_nums, residuals, "b-o", linewidth=2, markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_vs_stepsize(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    order: Optional[float] = None,
    title: str = "Error vs Step Size",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot error vs step size on log-log scale.

    Parameters
    ----------
    step_sizes : ndarray
        Array of step sizes.
    errors : ndarray
        Corresponding errors.
    order : float, optional
        Expected convergence order (plots reference line).
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(step_sizes, errors, "r-s", linewidth=2, markersize=6, label="Numerical error")

    if order is not None:
        # Plot reference line with slope = -order
        h_ref = step_sizes
        error_ref = errors[0] * (h_ref / step_sizes[0]) ** order
        ax.loglog(h_ref, error_ref, "k--", linewidth=1, label=f"O(h^{order})")

    ax.set_xlabel("Step Size h")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_ode_solution(
    t: np.ndarray,
    y: np.ndarray,
    t_exact: Optional[np.ndarray] = None,
    y_exact: Optional[np.ndarray] = None,
    title: str = "ODE Solution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot ODE solution trajectory.

    Parameters
    ----------
    t : ndarray
        Time points.
    y : ndarray
        Solution values (can be multi-dimensional).
    t_exact : ndarray, optional
        Exact time points.
    y_exact : ndarray, optional
        Exact solution values.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if y.ndim == 1:
        ax.plot(t, y, "b-", linewidth=2, label="Numerical")
        if y_exact is not None and t_exact is not None:
            ax.plot(t_exact, y_exact, "r--", linewidth=2, label="Exact")
    else:
        for i in range(min(y.shape[1], 5)):  # Plot up to 5 components
            ax.plot(t, y[:, i], linewidth=2, label=f"Component {i+1}")

    ax.set_xlabel("Time t")
    ax.set_ylabel("Solution y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
