# NumAnalytica Release History

## [0.2.0] - 2026-06-23

### Major Features
- **Backward Differentiation Formulas (BDF)** for stiff ODEs with A-stability analysis
- **Complex Step Differentiation (CSD)** for machine-precision Jacobians (error ~10⁻¹⁵)
- **Implicit ODE Solvers**: Backward Euler (BDF-1, A-stable)
- **Root-Finding Methods**: Newton-Raphson, Bisection, Secant, False Position, Fixed Point, Müller
- **Stability Analysis**: Region visualization and A-stability checking
- **Pedagogical Design**: Full iteration logging for educational transparency ("glass-box" approach)

### Bugs Fixed (v0.2.0 Cleanup)
- **NewtonRaphson.solve**: `bracket` parameter now works independently when `x0=None` (previously ignored)
- **BackwardEuler.solve**: `step_sizes` now correctly logs actual step size per iteration instead of hardcoding `[h] * iteration`
- **Jacobian Conditioning**: Fixed variable naming (`is_illconditioned` → `is_ill_conditioned`) for correct dict key generation

### Code Quality Improvements
- Corrected exception name: `BrackettingError` → `BracketingError`
- Deferred matplotlib imports in `stability/region_plotter.py` (eliminates heavy import overhead for solver-only users)
- Simplified README install instructions (removed unnecessary `cd src` step)
- Updated project URLs in pyproject.toml to match GitHub repository (A-MIKs/numanalytica)

### API Stability
- 43 public exports maintained across core, differentiation, roots, ODE, and stability modules
- All mathematical methods verified correct (9 mathematical tests passing)
- Runtime robustness confirmed (no crashes, proper error handling)

---

## [0.1.0] - 2026-04-01 (Initial Release)

### Core Features Implemented
- **BaseSolver**: Abstract base class for all solvers with unified logging
- **ODE Integrators**:
  - ExplicitEuler (forward Euler, non-stiff baseline)
  - BackwardEuler (implicit, A-stable for stiff systems)
- **Root Finding**:
  - Newton-Raphson with Complex Step Differentiation support
  - Bisection, Secant, False Position, Fixed Point, Müller methods
- **Differentiation**:
  - Complex Step Differentiation (CSD): error ~5.55e-17 vs finite difference ~2.43e-10
  - Finite differences (forward, backward, centered)
  - JacobianComputer factory with multiple strategy support
- **Stability Analysis**:
  - Stability regions for Forward Euler, Backward Euler, BDF-2, RK4, Midpoint
  - A-stability verification utilities
  - Dahlquist test problem support

### Initial Mathematical Issues (Fixed)
- **BDF2 characteristic polynomial**: Corrected from ζ² - (4/3 + (2/3)z)ζ + 1/3 = 0 to correct form ζ²(1 - (2/3)z) - (4/3)ζ + 1/3 = 0
- **BDF2 root selection**: Changed from min (best-case) to max (worst-case) for correct A-stability boundaries
- **Complex Step Jacobian dtype bug**: Added `.astype(complex)` before adding imaginary perturbation (critical for system solvers)
- **RootResult printing crash**: Made `__str__()` None-safe for failed convergence
- **Floating-point loop termination**: Guard changed from fixed `1e-14` to proportional `tf - h*0.5` for robust multi-step integration
- **setup.py syntax error**: Replaced broken code with minimal valid shim

### Test Coverage (v0.1.0)
- CSD accuracy: 7 orders of magnitude improvement over finite differences
- Newton-Raphson: 5 iterations to convergence on test cubic, quadratic convergence verified
- Backward Euler: Converged on Van der Pol μ=1 over [0,10] with 100 steps
- BDF2 stability: Correctly identifies stable/unstable regions on complex plane

### Documentation
- Comprehensive README with quick-start examples
- Inline code documentation for all public APIs
- Thesis action plan tracking
- Architecture documentation

---

## Development Timeline

### Phase 1: Initial Audit & Critical Fixes (April 1-15, 2026)
- Comprehensive codebase audit identifying 9+ issues across 3 severity tiers
- Fixed 5 mathematical bugs affecting core solver correctness
- CSD dtype fix enabling accurate Jacobian computation for systems
- All critical blockers resolved

### Phase 2: BDF & Stability Analysis (April 15-May 15, 2026)
- BDF-2 characteristic polynomial correction (fundamental math error)
- Root selection strategy verified for multistep stability
- Stability region visualization and A-stability analysis tools
- Dahlquist test utilities for stiff problem testing

### Phase 3: Robustness & Integration (May 15-June 1, 2026)
- Floating-point accumulation guard fixed for 100+ step integrations
- Newton-Raphson system solver with reusable LU decomposition
- Backward Euler implicit step solver with accurate convergence tracking
- End-to-end integration tests passing

### Phase 4: Publication Preparation (June 1-23, 2026)
- GitHub repository initialized and pushed
- README restructured for GitHub display
- Cleanup phase: 7 quality improvements and bug fixes
- Comprehensive test suite for all fixes
- Ready for thesis submission and PyPI publication

---

## Version Roadmap (Future)

### [0.3.0] (Planned)
- Full BDF multistep family (BDF-3 through BDF-6)
- Adaptive step-size control
- Implicit-explicit (IMEX) integrators
- Extensive pytest test suite
- Performance benchmarking suite

### [1.0.0] (Planned - Thesis Release)
- Comprehensive documentation with mathematical background
- Publication-ready code with full type hints
- PyPI release
- Academic paper reference and citation guidelines

---

## Known Limitations (v0.2.0)

1. **Single-step vs Multistep**: BDF-2 exported as stability analysis only; no dedicated BDF-2 solver class
2. **Step Size Control**: Fixed step size only; no adaptive stepping
3. **Problem Scope**: Designed for educational/pedagogical use; SciPy recommended for production use
4. **Visualization**: matplotlib required for stability region plots (can be deferred import)

---

## Citation

If you use NumAnalytica in academic work, please cite:

```
Kola-Ilugbo Ayomikun (2026). 
"Implementation and Stability Analysis of Backward Differentiation Formulas 
for Stiff Differential Equations: The NumAnalytica Library Approach."
B.Sc. Thesis, [University Name].
```

---

## Contributors

- **Kola-Ilugbo Ayomikun** — Project creator and lead developer
- **GitHub Copilot** — Code review, debugging, and optimization assistance

---

## License

MIT License — See LICENSE file for details.
