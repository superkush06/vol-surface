# Contributing

## Run what CI runs

```bash
pip install -e ".[dev,plot]"
make check          # ruff, mypy --strict, pytest, examples/validate.py
```

`make check` is the same four commands in the same order as
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), so a green `make check`
locally is a green build. `pre-commit install` puts ruff and the whitespace
hooks in front of every commit.

## What a change needs

- **A test with an oracle.** The suite is built on comparisons against
  something outside the code: a closed form transcribed from the source paper,
  a limit the model must collapse to, a Monte Carlo of the SDE the formula
  approximates, or a second screen sharing no code with the first. A test that
  asserts the current output is not evidence; it is a snapshot.
- **Every printed number rechecked.** `tests/test_readme.py` recomputes the
  literals in `README.md` and `tests/test_validation.py` does the same for
  `docs/validation.md`. If your change moves a number, the suite tells you
  which, and the document is what you fix.
- **A CHANGELOG entry**, under a new heading if it is a release.
- **Type annotations.** The package ships `py.typed`, and `mypy --strict`
  passes over `volsurf/`, and new code has to keep that true.

## Scope

This library turns a screen of quotes into a volatility at a strike and expiry
nobody listed, and stops there. Pricing contracts, simulating paths, and
aggregating risk belong to other libraries.

Things that are unlikely to be merged, so you know before you write them:

- **A scipy, pandas, or numpy dependency in `volsurf/`.** The package imports
  the standard library only. That is a constraint the code is designed around,
  not an accident. `numpy` appears in the `dev` and `plot` extras because
  `examples/validate.py` and `docs/figures.py` are scripts, not package code.
- **Another parameterisation for its own sake.** SSVI, eSSVI, and local
  volatility are all reasonable; each one needs to arrive with the validation
  that makes it trustworthy, which is a much larger piece of work than the fit.
- **Vectorisation that costs readability.** Everything is scalar Python and
  calibration is comfortable at that speed. If you need throughput, the honest
  answer is a different library, not a rewrite of this one.
- **Speed claims without a benchmark in the repo**, or accuracy claims without
  a reference in `docs/validation.md`.

Bug reports are more useful with the numbers attached: the panel or quote
screen that reproduces it, your OS, and `python -c "import volsurf; print(volsurf.__version__)"`.
