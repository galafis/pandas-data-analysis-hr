.PHONY: install run test lint data clean docker-build docker-run

PYTHON := python3
PIP := pip

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.data_loader

run:
	$(PYTHON) -m src.pipeline

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage

docker-build:
	docker-compose build

docker-run:
	docker-compose up pipeline

docker-test:
	docker-compose run --rm test

help:
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  data         Download/generate dataset"
	@echo "  run          Run full pipeline"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Lint with ruff"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run pipeline in Docker"
