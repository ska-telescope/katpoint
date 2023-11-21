PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88

include .make/base.mk
include .make/python.mk


docs-pre-build:
	git fetch --unshallow
	poetry install --with docs

python-pre-lint:
	git fetch --unshallow
