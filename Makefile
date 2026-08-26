.PHONY: up down build logs test test-fast

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f backend

test:
	python -m pytest backend/tests/ -v

test-fast:
	python -m pytest backend/tests/ -v -m "not slow"

lint:
	python -m pre_commit run --all-files
