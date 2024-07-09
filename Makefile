PROJECT = katpoint
PROJECT_NAME = katpoint
PYTHON_LINE_LENGTH = 88

include .make/base.mk
include .make/python.mk


# XXX Reinstall Poetry environment so that dynamic versioning can take effect
python-pre-lint:
	poetry install

python-pre-test:
	poetry install

# Test that we actually have a working package after a pip install
python-post-test:
	pip3 install .
	python3 -c "import katpoint"

python-pre-build:
	poetry install

python-post-build:
	pip3 install twine
	twine check ./dist/*

# XXX Reinstall Poetry environment so that dynamic versioning can take effect.
# This has to be done before the `--with docs` step that needs a proper version.
# XXX Also install package itself with Poetry to access `katpoint.__version__`
# in docs (unlike the default CI step that has `--no-root --only docs` options).
docs-pre-build:
	poetry install
	poetry install --with docs
