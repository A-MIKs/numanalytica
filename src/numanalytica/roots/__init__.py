"""Root-finding module: Newton-Raphson, Bisection, Secant, False Position, Fixed-Point, Müller."""

from numanalytica.roots.bisection import Bisection
from numanalytica.roots.false_position import FalsePosition
from numanalytica.roots.fixed_point import FixedPoint
from numanalytica.roots.muller import Muller
from numanalytica.roots.newton_raphson import NewtonRaphson, NewtonRaphsonSystem
from numanalytica.roots.secant import Secant
from numanalytica.roots.utils import (
    check_bracket_validity,
    get_initial_interval,
    refine_bracket,
)

__all__ = [
    "NewtonRaphson",
    "NewtonRaphsonSystem",
    "Bisection",
    "Secant",
    "FalsePosition",
    "FixedPoint",
    "Muller",
    "get_initial_interval",
    "check_bracket_validity",
    "refine_bracket",
]
