# Changelog

## [0.1.0] - 2026-07-XX

### Added
- Black-Scholes pricing + delta, gamma, vega.
- Implied volatility solver using Brent's method (pure Python).
- SABR Hagan-formula IV with parameter validation.
- 3-parameter SABR slice calibration (alpha, rho, nu) at fixed beta —
  grid-search init + Nelder-Mead, scipy-free.
- SVI raw parameterisation with total-variance / IV conversion.
- No-arbitrage checks: butterfly density + calendar monotonicity.
- CI on Python 3.11 + 3.12.
