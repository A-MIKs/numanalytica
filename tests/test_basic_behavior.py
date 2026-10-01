import numpy as np
import pytest

import numanalytica.roots.newton_raphson as nr_mod
from numanalytica import BackwardEuler, NewtonRaphson
from numanalytica.core.results import IntegrationResult


def test_newton_raphson_converges():
    def f(x):
        return x**2 - 4

    result = NewtonRaphson(f, verbose=False).solve(x0=1.0, tol=1e-9)

    assert result.converged
    assert result.root is not None
    assert abs(result.root - 2.0) < 1e-8


def test_backward_euler_converges():
    def rhs(t, y):
        return -y

    result = BackwardEuler(rhs, verbose=False).solve(
        t0=0.0,
        tf=1.0,
        y0=np.array([1.0]),
        h=0.25,
    )

    assert result.converged
    assert result.y.shape[0] >= 2
    expected = np.array([(1 / 1.25) ** 4])
    assert np.allclose(result.y[-1], expected, atol=1e-12)


def test_integration_result_handles_empty_time_grid():
    result = IntegrationResult(
        solution=None,
        converged=False,
        iterations=0,
        residual=1.0,
        tolerance=1e-6,
        message="failed",
        elapsed_time=0.0,
    )

    text = str(result)
    assert "ODE Integration Result" in text
    assert "Status:" in text


def test_backward_euler_rejects_non_positive_step():
    solver = BackwardEuler(lambda t, y: -y, verbose=False)

    with pytest.raises(ValueError, match="h must be positive"):
        solver.solve(t0=0.0, tf=1.0, y0=np.array([1.0]), h=0.0)


def test_backward_euler_respects_t_eval():
    solver = BackwardEuler(lambda t, y: -y, verbose=False)

    result = solver.solve(
        t0=0.0,
        tf=1.0,
        y0=np.array([1.0]),
        t_eval=np.array([0.0, 0.5, 1.0]),
    )

    assert np.allclose(result.t, np.array([0.0, 0.5, 1.0]))
    assert result.y.shape == (3, 1)
    assert np.allclose(result.y[-1], np.array([1.0 / 1.5**2]), atol=1e-12)


def test_newton_raphson_uses_requested_derivative_method(monkeypatch):
    def f(x):
        return x**2 - 4

    def fake_complex_step_derivative(*args, **kwargs):
        raise AssertionError("complex-step derivative should not be used when method='finite_diff'")

    def fake_finite_difference_centered(g, x, h=1e-8, args=()):
        return (g(x + h, *args) - g(x - h, *args)) / (2 * h)

    monkeypatch.setattr(nr_mod, "complex_step_derivative", fake_complex_step_derivative)
    monkeypatch.setattr(nr_mod, "finite_difference_centered", fake_finite_difference_centered)

    result = NewtonRaphson(f, method="finite_diff", verbose=False).solve(x0=2.0, tol=1e-9)

    assert result.converged
    assert abs(result.root - 2.0) < 1e-6


def test_newton_raphson_zero_maxiter_is_safe():
    result = NewtonRaphson(lambda x: x**2 - 4, verbose=False).solve(x0=2.0, maxiter=0)

    assert not result.converged
    assert result.iterations == 0
