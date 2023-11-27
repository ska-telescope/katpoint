PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88
CI_POETRY_VERSION = 1.7.1

include .make/base.mk
include .make/python.mk


python-pre-lint:
	poetry dynamic-versioning

python-pre-test:
	poetry dynamic-versioning

python-pre-build:
	poetry dynamic-versioning

docs-pre-build:
	poetry self add "poetry-dynamic-versioning[plugin]"
	poetry install --with docs
