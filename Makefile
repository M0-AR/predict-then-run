.PHONY: install freeze run score readme test lint docker-build docker-run clean

install:
	pip install -e ".[dev]"

freeze:
	python scripts/freeze.py

run:
	python scripts/run_all.py --seeds 0,1,2,3,4 --steps 300

score:
	python scripts/score.py

readme:
	python scripts/render_readme.py

test:
	pytest -q

lint:
	ruff check src scripts tests

docker-build:
	docker compose build

docker-run:
	docker compose run --rm benchmark

clean:
	rm -rf results/raw/*.json results/scores.json results/tables/*.csv
