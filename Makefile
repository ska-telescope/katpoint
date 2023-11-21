PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88
CI_POETRY_VERSION = 1.7.1

include .make/base.mk
include .make/python.mk


docs-pre-build:
	poetry install --with docs

python-pre-scan:
	pip install poetry==${CI_POETRY_VERSION} poetry-dynamic-versioning[plugin]
