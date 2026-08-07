# Changelog

## [0.6.0] - 2026-08-07

### Added
- `mypy --strict` over `volsurf`, wired into CI. The annotations it needed
  were mostly on inner closures (`implied_vol`'s objective, `_brentq`'s
  callable parameter) rather than on the public surface, which was already
  typed.
- A browser demo under `docs/demo`, published at
  https://superkush06.github.io/vol-surface/demo/. It runs this package under
  Pyodide: a rotatable fitted surface, a slice whose five SVI parameters you
  can push until the implied density goes negative, calibration against noisy
  quotes, and SABR scored on the same marks. `docs/demo/vs.py` imports the
  truth surface and the noise from `examples/fit_surface.py`, so the residuals
  the page reports cannot drift from the ones the README quotes.

### Changed
- CI runs on macOS as well as Linux. `volsurf` pins printed digits in its
  README, and glibc and Apple libm are not required to agree to the last ulp
  on `erf`/`exp`/`log`, so a single-platform matrix was checking that the
  numbers were reproducible rather than that they were right.
- `sabr_iv` computes `(F*K)**((1-beta)/2)` through `math.pow`, which is the
  same value with a defined type for a strict checker.
- numpy is pinned below 3 in the `dev` and `plot` extras.

### Fixed
- Three README numbers were pinned to display precision on values that come
  out of the optimiser, and the new matrix failed on two of them before finding
  something worse. The repaired-smile claim read `at most 1.83 vol points` and
  CPython 3.11 computes 1.81; the recovery claim pinned the first decimal of a
  relative error and was one tweak from the same break. Both are bounds now.
- The soft-penalty caveat was wrong, not merely over-precise. It claimed the
  penalty leaves `min g(k)` above zero, and on macOS with CPython 3.11 the same
  source lands `-8.3e-05`. `butterfly_penalty` adds a term to the objective
  rather than constraining the feasible set, so the sign was never guaranteed,
  which is what "not a projection" in that same sentence already said. The
  claim is now what the penalty actually delivers: `|min g(k)|` pulled from
  `-0.0328` to within `1e-4`, better than two orders of magnitude, with the
  sign left to the optimiser's path.
- `test_the_readme_states_the_real_test_count` cleared the repository's own
  `-ra -q` addopts before collecting. Passing `-q` on top of it made pytest
  doubly quiet and dropped the "N tests collected" line the test parses.

## [0.5.0] - 2026-07-27

### Added
- `docs/validation.md` — the library checked against things it does not
  control: Hagan's own at-the-money formula, the Black and normal limits, a
  conditional Monte-Carlo of the SABR SDE, Lee's wing-slope bound, the
  Gatheral-Jacquier sufficient conditions on SSVI, and the discrete price
  screen against the analytic one. Includes the places where the two sides
  disagree and why.
- `examples/validate.py` prints every number in that document; CI runs it.
- `tests/test_validation.py` asserts the same comparisons at smaller sample
  sizes, importing the reference implementations from `examples/validate.py`
  so the doc and the suite cannot drift apart.
- `tests/test_properties.py` — seeded randomised invariant tests: put-call
  parity, static bounds, Greeks against differences of the price,
  Breeden-Litzenberger against the lognormal density, SABR's scale invariance
  and beta=0 reflection symmetry, continuity through the money, the SVI
  density's unit mass and martingale condition, the g/density sign agreement,
  the wing limit of g, calibration recovery on random draws, and both
  discrete screens.
- `examples/greeks_handoff.py` — a quote screen through calibration and the
  arbitrage screens to a marked book, its Greeks, and a scenario P&L vector,
  which is the shape of the handoff to the pricing and risk libraries next to
  this one.
- numpy is now a `dev` extra (it was already a `plot` extra); the package
  itself still imports only the standard library.

### Fixed
- `implied_vol` returns the bracket end when the market price sits exactly on
  it. Deep out of the money the price is flat in sigma to the last bit, so
  `f(lo)` is exactly zero and Brent rejected the bracket with an internal
  "f(a)*f(b) >= 0" message. Found by the randomised round-trip test.
- `sabr_fit._objective` zips its three inputs with `strict=True`, so a length
  mismatch raises instead of silently fitting a prefix.

### Changed
- README: the `svi_min_g` output and the `fit_surface.py` residual range now
  match what a rerun prints, and there are two new sections — what has been
  checked against what, and where this library sits next to the others.

## [0.4.0] - 2026-07-27

### Added
- `svi_min_g(params, k_min, k_max)` returns the smallest g(k) **and where it
  occurs** — knowing which strikes are implicated is the point of running the
  check, and a bare boolean throws that away.
- `SVISurface.butterfly_arbitrage_free(k_min, k_max)`, the per-slice
  companion to `calendar_arbitrage_free`. A surface needs both.
- `docs/figures.py` renders every figure in the README from live library
  output: the butterfly diagnosis and its repair, the calibration
  comparison, and the fitted surface with its calendar condition.
- `examples/fit_surface.py` — five expiries, 21 strikes each, 25 bp of noise
  on every quote, through to a vol at an expiry nobody quoted.
- `examples/screen_for_arbitrage.py` — the three screens (analytic
  butterfly, discrete butterfly, calendar) on data that trips each one.

### Changed
- README rewritten around what the library is for rather than what it
  contains, with every printed number taken from an actual run.
- `docs/theory.md` expanded into a real derivation: Breeden-Litzenberger,
  the near-ATM series for z/x(z), where g(k) comes from, why the
  quasi-explicit reduction removes the local minimum, and full citations.
  Formulas now use `$...$`, which GitHub renders — the old `\[...\]` did not.
- `examples/fit_sabr.py` also walks the strike towards the forward, showing
  the ATM value reached as a limit rather than a branch.
- `examples/fit_svi.py`, `examples/svi_density.py` and
  `examples/render_hero.py` are folded into the new examples and
  `docs/figures.py`; `docs/demo.png` and `docs/density.png` are replaced by
  `docs/hero.png`, `docs/surface.png` and `docs/calibration.png`.

## [0.3.0] - 2026-07-24

### Added
- **Gatheral-Jacquier butterfly machinery**: `svi_g` (closed-form g(k) for
  raw SVI), `svi_density` (implied risk-neutral density of log-moneyness),
  `svi_butterfly_arbitrage_free`, and `ButterflyArbitrageWarning`.
- `fit_svi_slice` now checks its own output for butterfly arbitrage
  (warns when min g(k) < 0 on the fit range) and accepts an optional
  `butterfly_penalty` that pushes the fit toward arbitrage-free parameters.
- `examples/svi_density.py` — renders the Vogt slice's g(k) and its
  negative implied density (`docs/density.png`).

### Changed
- **`fit_svi_slice` rewritten** around the quasi-explicit De Marco/Zeliade
  reduction: for fixed (m, sigma) the remaining SVI parameters solve a 3x3
  linear system exactly, so only a 2-D outer search (+ multistart + 5-D
  polish) remains. Recovers steep skews the old single-start 5-D
  Nelder-Mead missed by ~19%.
- `butterfly_violations` now requires an explicit `forward` and prices all
  strikes at it. The old scan priced each strike triple at its own middle
  strike, which provably misses genuine butterflies.
- `fit_svi_surface` forwards keyword arguments to `fit_svi_slice`.

### Fixed
- `sabr_iv` no longer has a ~2% vol discontinuity in a band just off the
  money: the near-ATM z/x(z) factor is evaluated by Taylor series and the
  time-correction factor is applied on every path (ATM is now the smooth
  limit of one code path, not a special case).
- `SVISurface.total_variance` scales the first slice proportionally in T
  below the shortest fitted expiry, so short-end implied vol stays at the
  first slice's level instead of diverging as T -> 0.
- `calendar_violations` raises on mismatched input lengths instead of
  silently truncating the scan.
- `implied_vol` now says which bracket end the vol fell outside (e.g.
  "exceeds the upper bracket hi=5.0") instead of a bare "no sign change",
  and no longer evaluates the objective twice at the bracket ends.

## [0.2.0] - 2026-06-21

### Added
- **Multi-expiry SVI surface**: `fit_svi_slice` (scipy-free least-squares
  raw-SVI fit), `SVISurface` with linear-in-T total-variance interpolation,
  `fit_svi_surface`, and a `calendar_arbitrage_free` check.
- **3-D vol-surface hero chart** (`examples/render_hero.py` → `docs/demo.png`):
  the implied-vol surface plus per-expiry smiles.

## [0.1.0] - 2026-06-18

### Added
- Black-Scholes pricing + delta, gamma, vega.
- Implied volatility solver using Brent's method (pure Python).
- SABR Hagan-formula IV with parameter validation.
- 3-parameter SABR slice calibration (alpha, rho, nu) at fixed beta —
  grid-search init + Nelder-Mead, scipy-free.
- SVI raw parameterisation with total-variance / IV conversion.
- No-arbitrage checks: butterfly density + calendar monotonicity.
- CI on Python 3.11 + 3.12.
