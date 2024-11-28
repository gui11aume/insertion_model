PYTHON_VERSION := 3.10.12

POETRY := PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry
VENV := .venv
DEV_MARKER := .dev-dependencies-installed
DOCKER_BUILD_MARKER := .docker-build-installed
DOCKER_CONTAINER_NAME := $(shell awk -F'"' '/^name = / {print substr($$2, 1)}' pyproject.toml | sed 's/-/_/g')
PRECOMMIT_MARKER := .pre-commit-installed
PYTHON_INSTALLED_MARKER := .python-$(PYTHON_VERSION)-installed

.PHONY: install install-dev clean train test test-ci


train: install
	$(POETRY) run python demo_train.py \
		--config config.yaml \
		--output trained_model_weights.pt

download-data: install
	$(POETRY) run python demo_train.py \
		--config config.yaml \
		--download-only

$(PYTHON_INSTALLED_MARKER):
	pyenv install -s $(PYTHON_VERSION)
	touch $(PYTHON_INSTALLED_MARKER)

$(VENV): pyproject.toml $(PYTHON_INSTALLED_MARKER)
	pyenv local $(PYTHON_VERSION)
	$(POETRY) env use $$(pyenv which python)
	touch $(VENV)

install: $(VENV)
	$(POETRY) install --only main

$(DEV_MARKER): $(VENV)
	$(POETRY) install --with dev
	touch $(DEV_MARKER)

install-dev: $(DEV_MARKER) $(PRECOMMIT_MARKER)

$(PRECOMMIT_MARKER): $(DEV_MARKER)
	$(POETRY) run pip install pre-commit
	$(POETRY) run pre-commit install
	touch $(PRECOMMIT_MARKER)

clean:
	rm -rf $(VENV)
	rm -f .python-version $(PRECOMMIT_MARKER) $(PYTHON_INSTALLED_MARKER) $(DEV_MARKER)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


# Run tests using pytest
test: $(DEV_MARKER)
	$(POETRY) run pytest tests/ $(ARGS)

# Run tests using pytest and generate reports
test-ci: $(DEV_MARKER)
	$(POETRY) run pytest tests/ $(ARGS) \
		--cov=src \
		--cov-branch \
		--cov-report=term \
		--cov-report=term-missing \
		--cov-report=lcov:coverage.lcov \
		--junitxml=pytest.xml \

# Run a single file (demo)
test-one-file: $(DEV_MARKER)
	$(POETRY) run pytest tests/test_demo.py

# Run a single case (demo)
test-one-case: $(DEV_MARKER)
	$(POETRY) run pytest tests/test_demo.py::test_something

# Docker
docker-build: poetry.lock
	$(POETRY) export --without-hashes -o docker/requirements.txt
	docker build -f docker/Dockerfile -t $(DOCKER_CONTAINER_NAME) .
	touch $(DOCKER_BUILD_MARKER)

docker-run: $(DOCKER_BUILD_MARKER)
	docker run --rm --gpus all \
		-v $$(pwd)/datasets:/workspace/datasets \
		-v $$(pwd)/config.yaml:/workspace/config.yaml \
		$(DOCKER_CONTAINER_NAME) \
		python3 demo_train.py --config config.yaml --output trained_model_weights.pt
