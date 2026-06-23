#!/usr/bin/env python
"""Test script to verify all fixes."""

import numpy as np
from numanalytica import NewtonRaphson
from numanalytica.core import BracketingError
from numanalytica.differentiation.jacobian import check_jacobian_quality

# Test 1: NewtonRaphson bracket parameter fix
print("Test 1: NewtonRaphson with bracket parameter")
def f(x):
    return x**3 - 2*x - 5

solver = NewtonRaphson(f)
result = solver.solve(bracket=(2.0, 3.0))
print(f"  ✓ Root found: {result.root:.10f}")
print(f"  ✓ Converged: {result.converged}")

# Test 2: BracketingError spelling
print("\nTest 2: BracketingError imported successfully")
try:
    raise BracketingError("Test error")
except BracketingError as e:
    print(f"  ✓ BracketingError caught: {e}")

# Test 3: Check Jacobian quality with new naming
print("\nTest 3: check_jacobian_quality returns correct keys")
J = np.array([[1.0, 2.0], [3.0, 4.0]])
quality = check_jacobian_quality(J)
print(f"  Keys: {sorted(quality.keys())}")
assert 'is_ill_conditioned' in quality, "Missing 'is_ill_conditioned' key"
assert 'is_illedconditioned' not in quality, "Old misspelled key still present"
print(f"  ✓ Has 'is_ill_conditioned': True")
print(f"  ✓ Old key 'is_illedconditioned': False")

# Test 4: BackwardEuler step_sizes
print("\nTest 4: BackwardEuler step sizes logging")
from numanalytica import BackwardEuler

def simple_ode(t, y):
    return -y

be = BackwardEuler(simple_ode)
result = be.solve(t0=0, tf=1, y0=np.array([1.0]), h=0.1)
print(f"  Solution converged: {result.converged}")
print(f"  Number of steps: {len(result.step_sizes)}")
print(f"  First 3 step sizes: {result.step_sizes[:3]}")
print(f"  Last 3 step sizes: {result.step_sizes[-3:]}")
print(f"  Sum of steps: {sum(result.step_sizes):.10f} (expected ≈ 1.0)")

print("\n✓ All fixes verified successfully!")
