"""Benchmarking module: Van der Pol oscillator and performance comparison."""

import numpy as np


def van_der_pol(t, y, mu=1.0):
    """
    Van der Pol oscillator: a standard benchmark for stiff/non-stiff testing.

    ODE system:
        dy1/dt = y2
        dy2/dt = μ(1 - y1²)*y2 - y1

    Parameters
    ----------
    t : float
        Time (not used in autonomous system).
    y : ndarray
        State vector [y1, y2].
    mu : float, default=1.0
        Stiffness parameter. Larger μ → stiffer problem.

    Returns
    -------
    ndarray
        Derivative [dy1/dt, dy2/dt].
    """
    y1, y2 = y[0], y[1]
    dy1 = y2
    dy2 = mu * (1 - y1**2) * y2 - y1
    return np.array([dy1, dy2])


def van_der_pol_jacobian(t, y, mu=1.0):
    """
    Jacobian of Van der Pol oscillator.

    J = [[0, 1], [-1-2μy1*y2, μ(1-y1²)]]

    Parameters
    ----------
    t : float
        Time.
    y : ndarray
        State [y1, y2].
    mu : float
        Stiffness parameter.

    Returns
    -------
    ndarray
        Jacobian matrix (2, 2).
    """
    y1, y2 = y[0], y[1]
    J = np.array([[0, 1], [-1 - 2 * mu * y1 * y2, mu * (1 - y1**2)]])
    return J


def conductance_model(t, y, g_na=120, g_k=36, g_l=0.3, v_na=115, v_k=-12, v_l=10.6, i_ext=0):
    """
    Hodgkin-Huxley model (simplified): physiological stiff system.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray
        State variables.
    g_na, g_k, g_l : float
        Conductances.
    v_na, v_k, v_l : float
        Nernst potentials.
    i_ext : float
        External current stimulus.

    Returns
    -------
    ndarray
        System derivative.
    """
    # Simplified version for demonstration
    v = y[0]
    dv = i_ext - g_na * (v - v_na) - g_k * (v - v_k) - g_l * (v - v_l)
    return np.array([dv])


class BenchmarkProblem:
    """Encapsulates a benchmark ODE problem with metadata."""

    def __init__(
        self,
        name: str,
        f,
        jacobian=None,
        y0: np.ndarray = None,
        t0: float = 0.0,
        tf: float = 1.0,
        stiffness_ratio: float = None,
    ):
        """
        Parameters
        ----------
        name : str
            Problem name.
        f : callable
            RHS function.
        jacobian : callable, optional
            Jacobian function.
        y0 : ndarray
            Initial condition.
        t0, tf : float
            Time range.
        stiffness_ratio : float, optional
            Ratio of fastest to slowest time scales (stiffness indicator).
        """
        self.name = name
        self.f = f
        self.jacobian = jacobian
        self.y0 = y0 if y0 is not None else np.ones(1)
        self.t0 = t0
        self.tf = tf
        self.stiffness_ratio = stiffness_ratio


# Library of benchmark problems
BENCHMARK_LIBRARY = {
    "Van der Pol (μ=1, non-stiff)": BenchmarkProblem(
        name="Van der Pol (μ=1)",
        f=lambda t, y: van_der_pol(t, y, mu=1.0),
        jacobian=lambda t, y: van_der_pol_jacobian(t, y, mu=1.0),
        y0=np.array([1.0, 0.0]),
        t0=0.0,
        tf=10.0,
        stiffness_ratio=10,
    ),
    "Van der Pol (μ=100, stiff)": BenchmarkProblem(
        name="Van der Pol (μ=100, STIFF)",
        f=lambda t, y: van_der_pol(t, y, mu=100.0),
        jacobian=lambda t, y: van_der_pol_jacobian(t, y, mu=100.0),
        y0=np.array([1.0, 0.0]),
        t0=0.0,
        tf=300.0,
        stiffness_ratio=1e6,
    ),
}
