"""
Stability analysis for numerical ODE methods.

This module computes and visualizes the stability regions of various numerical
integration methods in the complex plane. Stability analysis is crucial for
understanding method suitability for stiff vs. non-stiff problems.

Theory:
    For the test equation dy/dt = λy, a method is A-stable if its stability
    region {z ∈ ℂ : |ρ(z)| ≤ 1} covers the entire left half-plane Re(z) < 0.

    where ρ(z) is the stability function (amplification factor).
"""

from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


class StabilityRegion:
    """
    Computes and analyzes the stability region of a numerical method.

    Parameters
    ----------
    stability_func : callable
        Function ρ(z) returning the amplification factor at complex point z.
        Signature: ρ(z: complex) -> complex
    name : str
        Name of the method (e.g., "Forward Euler").
    """

    def __init__(
        self,
        stability_func: Callable[[complex], complex],
        name: str = "Method",
    ):
        """Initialize StabilityRegion."""
        self.stability_func = stability_func
        self.name = name
        self._cached_region = None

    def is_in_stability_region(
        self,
        z: complex,
        tol: float = 1e-10,
    ) -> bool:
        """
        Check if a point is in the stability region.

        Parameters
        ----------
        z : complex
            Test point in the complex plane.
        tol : float, default=1e-10
            Tolerance for stability check |ρ(z)| ≤ 1.

        Returns
        -------
        bool
            True if |ρ(z)| ≤ 1 + tol.
        """
        try:
            rho_z = self.stability_func(z)
            return abs(rho_z) <= 1 + tol
        except (ValueError, RuntimeError, ZeroDivisionError):
            return False

    def compute_region(
        self,
        real_range: Tuple[float, float] = (-6, 2),
        imag_range: Tuple[float, float] = (-6, 6),
        resolution: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute stability region on a grid.

        Parameters
        ----------
        real_range : tuple, default=(-6, 2)
            Real axis range (Re axis).
        imag_range : tuple, default=(-6, 6)
            Imaginary axis range (Im axis).
        resolution : int, default=500
            Grid resolution (points per axis).

        Returns
        -------
        real_grid : np.ndarray
            Real axis values.
        imag_grid : np.ndarray
            Imaginary axis values.
        stability_matrix : np.ndarray
            Binary matrix (1 in stability region, 0 outside).
        """
        real = np.linspace(real_range[0], real_range[1], resolution)
        imag = np.linspace(imag_range[0], imag_range[1], resolution)

        stability_matrix = np.zeros((len(imag), len(real)))

        for i, im in enumerate(imag):
            for j, re in enumerate(real):
                z = complex(re, im)
                stability_matrix[i, j] = 1 if self.is_in_stability_region(z) else 0

        self._cached_region = (real, imag, stability_matrix)
        return real, imag, stability_matrix

    def plot_region(
        self,
        ax=None,
        real_range: Tuple[float, float] = (-6, 2),
        imag_range: Tuple[float, float] = (-6, 6),
        resolution: int = 500,
        colormap: str = "RdYlGn",
        show_left_half_plane: bool = True,
    ):
        """
        Plot the stability region.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        real_range, imag_range : tuple
            Range for axes.
        resolution : int
            Grid resolution.
        colormap : str
            Matplotlib colormap name.
        show_left_half_plane : bool
            If True, shade the left half-plane (where Re(z) < 0).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            The created figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        real, imag, stability = self.compute_region(
            real_range=real_range,
            imag_range=imag_range,
            resolution=resolution,
        )

        # Plot stability region
        extent = [real[0], real[-1], imag[0], imag[-1]]
        im = ax.imshow(
            stability,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap=colormap,
            alpha=0.7,
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Stability (1=stable, 0=unstable)", rotation=270, labelpad=15)

        # Mark imaginary axis
        ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-", label="Imaginary axis")
        ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-", label="Real axis")

        # Shade left half-plane
        if show_left_half_plane:
            ax.axvspan(real[0], 0, alpha=0.1, color="lightgray", label="Left half-plane")

        ax.set_xlabel("Re(hλ)", fontsize=12)
        ax.set_ylabel("Im(hλ)", fontsize=12)
        ax.set_title(f"Stability Region: {self.name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        return fig, ax


# Pre-built stability functions for common methods
def stability_forward_euler(z: complex) -> complex:
    r"""
    Stability function for Forward Euler: ρ(z) = 1 + z.

    Stability region is the disk |1 + z| ≤ 1.
    Not A-stable (doesn't contain left half-plane).
    """
    return 1 + z


def stability_backward_euler(z: complex) -> complex:
    r"""
    Stability function for Backward Euler: ρ(z) = 1 / (1 - z).

    Stability region is the disk |1 - z| ≥ 1 (complement).
    A-STABLE (contains entire left half-plane).
    """
    if abs(1 - z) < 1e-15:
        raise ZeroDivisionError("Pole at z=1")
    return 1 / (1 - z)


def stability_bdf2(z: complex) -> complex:
    r"""
    Stability function for BDF2 via characteristic polynomial.

    BDF2 applied to y' = λy, with z = hλ, gives the characteristic equation:
        ζ²(1 - (2/3)z) - (4/3)ζ + 1/3 = 0

    A point z is in the stability region iff max|ζ| ≤ 1
    over all roots of this polynomial.

    BDF2 is A(α)-stable with α ≈ 73.35° (almost but not fully A-stable).

    References: Dahlquist (1963), Butcher (2016) §5.5
    """
    # Quadratic coefficients: aζ² + bζ + c = 0
    a = 1 - (2 / 3) * z  # Leading coefficient (z-dependent!)
    b = -4 / 3  # Linear coefficient
    c = 1 / 3  # Constant term

    # Quadratic formula: discriminant and roots
    discriminant = b**2 - 4 * a * c
    sqrt_disc = discriminant**0.5
    root1 = (-b + sqrt_disc) / (2 * a)
    root2 = (-b - sqrt_disc) / (2 * a)

    # Stability requires ALL roots inside unit disk — return the worst (largest)
    return max([root1, root2], key=abs)


def stability_rk4(z: complex) -> complex:
    r"""
    Stability function for RK4: ρ(z) = 1 + z + z²/2 + z³/6 + z⁴/24.

    Fourth-order explicit method. Limited stability region.
    """
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def stability_midpoint(z: complex) -> complex:
    r"""
    Stability function for Midpoint rule: ρ(z) = (1 + z/2) / (1 - z/2).

    Second-order, A-stable.
    """
    if abs(1 - z / 2) < 1e-15:
        raise ZeroDivisionError("Pole at z=2")
    return (1 + z / 2) / (1 - z / 2)


# Build library of common methods
STABILITY_LIBRARY = {
    "Forward Euler": StabilityRegion(stability_forward_euler, "Forward Euler"),
    "Backward Euler": StabilityRegion(stability_backward_euler, "Backward Euler (BDF1)"),
    "BDF2": StabilityRegion(stability_bdf2, "BDF2"),
    "RK4": StabilityRegion(stability_rk4, "Runge-Kutta 4"),
    "Midpoint": StabilityRegion(stability_midpoint, "Midpoint Rule"),
}


class StabilityComparison:
    """
    Compare stability regions of multiple methods side-by-side.

    Parameters
    ----------
    methods : dict
        Dictionary {name: StabilityRegion}.
    """

    def __init__(self, methods: dict):
        """Initialize comparison."""
        self.methods = methods

    def plot_comparison(
        self,
        real_range: Tuple[float, float] = (-6, 2),
        imag_range: Tuple[float, float] = (-6, 6),
        resolution: int = 400,
    ):
        """
        Plot all stability regions in a grid.

        Parameters
        ----------
        real_range, imag_range : tuple
            Axis ranges.
        resolution : int
            Grid resolution.

        Returns
        -------
        fig, axes : matplotlib Figure and Axes
            The created figure and axes grid.
        """
        n_methods = len(self.methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for idx, (name, region) in enumerate(self.methods.items()):
            region.plot_region(
                ax=axes[idx],
                real_range=real_range,
                imag_range=imag_range,
                resolution=resolution,
            )

        # Remove extra subplots
        for idx in range(n_methods, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle("Stability Region Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig, axes[:n_methods]


def check_a_stability(region: StabilityRegion, n_test_points: int = 1000) -> float:
    """
    Check degree of A-stability for a method.

    Returns the fraction of the left half-plane covered by the stability region.

    Parameters
    ----------
    region : StabilityRegion
        The method's stability region.
    n_test_points : int
        Number of test points.

    Returns
    -------
    float
        Fraction in [0, 1] indicating A-stability degree.
        1.0 means fully A-stable.
    """
    # Sample left half-plane
    real_half = np.linspace(-10, 0, n_test_points // 2)
    imag_half = np.linspace(-10, 10, n_test_points // 2)

    count_in_region = 0
    for re in real_half:
        for im in imag_half:
            if region.is_in_stability_region(complex(re, im)):
                count_in_region += 1

    total = len(real_half) * len(imag_half)
    return count_in_region / total
