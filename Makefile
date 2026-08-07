# One command per task, and `make check` is exactly what CI runs.
PY ?= python

.PHONY: help install lint typecheck test check validate figures examples build clean

help:  ## list the targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*## /\t/' | expand -t 14

install:  ## editable install with the dev and plotting extras
	$(PY) -m pip install -e ".[dev,plot]"

lint:  ## ruff
	ruff check .

typecheck:  ## mypy --strict over volsurf (the package ships py.typed)
	mypy

test:  ## the test suite
	pytest

validate:  ## regenerate every number in docs/validation.md
	PYTHONPATH=. $(PY) examples/validate.py

check: lint typecheck test validate  ## everything CI runs, in CI's order

figures:  ## redraw docs/hero.png, calibration.png, surface.png from live output
	PYTHONPATH=. $(PY) docs/figures.py

examples:  ## run every script in examples/
	@for f in examples/*.py; do echo "== $$f"; PYTHONPATH=. $(PY) $$f >/dev/null || exit 1; done
	@echo "all examples ran"

build:  ## sdist + wheel into dist/
	$(PY) -m build

clean:  ## remove build products and tool caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
