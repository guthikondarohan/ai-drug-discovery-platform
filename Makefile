# Makefile for AI Drug Discovery Platform

.PHONY: help install test train api web docker-build docker-up docker-down clean

help:  ## Show this help message
	@echo "AI Drug Discovery Platform - Makefile Commands"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install black flake8 mypy isort

test:  ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov=api --cov-report=html

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

lint:  ## Run linters
	flake8 src api tests
	black --check src api tests
	isort --check-only src api tests

format:  ## Format code
	black src api tests
	isort src api tests

train:  ## Train the model
	python -m src.main --stage all --epochs 10

train-advanced:  ## Train with advanced models
	python -m src.main --stage all --epochs 20 --model ensemble

api:  ## Start FastAPI server
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

web:  ## Start Streamlit web app
	streamlit run chat_app.py

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start Docker containers
	docker-compose up -d

docker-down:  ## Stop Docker containers
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov

clean-models:  ## Remove trained models
	rm -rf results/*.pt

clean-data:  ## Remove processed data
	rm -rf data/processed/*.csv

rebuild:  ## Clean and rebuild everything
	make clean
	make install
	make train
	make test

.DEFAULT_GOAL := help
