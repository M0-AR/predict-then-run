FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY preregistry ./preregistry
COPY tests ./tests

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e ".[dev]"

CMD ["bash", "-lc", "ptr verify && python scripts/run_all.py --seeds 0,1,2,3,4 --steps 300 && python scripts/score.py && python scripts/render_readme.py"]
