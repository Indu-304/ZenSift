# Two stages: build the virtualenv with Poetry, then ship only the venv and source code.
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
# Runtime dependencies only; pytest and RAGAS stay out of the image.
RUN poetry install --only main --no-root


FROM python:3.12-slim

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ZENSIFT_DB_PATH=/app/data/zensift.db

RUN useradd --create-home --uid 1000 zensift
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config.yaml ./

RUN mkdir -p /app/data && chown zensift:zensift /app/data
USER zensift

# Mount a volume here so the sent-articles ledger survives between runs.
VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "src.main"]
