"""Visualization module for numerical results and diagnostics."""

from numanalytica.visualization.convergence_plot import (
    plot_convergence_history,
    plot_error_vs_stepsize,
    plot_ode_solution,
)

__all__ = [
    "plot_convergence_history",
    "plot_error_vs_stepsize",
    "plot_ode_solution",
]
