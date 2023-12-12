PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88
CI_POETRY_VERSION = 1.7.1

include .make/base.mk
include .make/python.mk


# XXX Reinstall Poetry so that dynamic versioning can take effect
python-pre-lint:
	poetry install

python-pre-test:
	poetry install

python-pre-build:
	poetry install

# XXX Install package itself with Poetry to access its dynamic version
docs-pre-build:
#	poetry self add "poetry-dynamic-versioning[plugin]"
	poetry install --with docs
