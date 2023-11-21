PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88

include .make/base.mk
include .make/python.mk


docs-pre-build:
	poetry install --with docs

python-pre-scan:
	pip install poetry==1.2.2 poetry-dynamic-versioning
