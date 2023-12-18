PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88
CI_POETRY_VERSION = 1.7.1

include .make/base.mk
include .make/python.mk


# XXX Reinstall Poetry environment so that dynamic versioning can take effect
python-pre-lint:
	poetry install

python-pre-test:
	poetry install

python-pre-build:
	poetry install

# XXX Reinstall Poetry environment so that dynamic versioning can take effect.
# This has to be done before the `--with docs` step that needs a proper version.
# XXX Also install package itself with Poetry to access `katpoint.__version__`
# in docs (unlike the default CI step that has `--no-root --only docs` options).
docs-pre-build:
	poetry install
	poetry install --with docs
