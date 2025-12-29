.PHONY: clean dist install dev release test test-cov lint format help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache .coverage htmlcov/ .ruff_cache
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

dist: clean ## Build source and wheel package
	uv build

install: ## Install package to current environment
	uv pip install -e .

dev: ## Install with dev dependencies
	uv sync --extra dev

test: ## Run tests
	uv run pytest -v

test-cov: ## Run tests with coverage
	uv run pytest --cov=hanzi_char_featurizer --cov-report=html --cov-report=term

lint: ## Check code style with ruff
	ruff check .

format: ## Format code with ruff
	ruff format .

release: dist ## Upload package to PyPI
	uv publish
