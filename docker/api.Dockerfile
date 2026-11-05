# ShopSage API (FastAPI + models + assistant). Data and fitted artifacts are mounted at runtime
# (build them on the host with `make data train eval`).
FROM python:3.12.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# pytensor (pymc-marketing) compiles small C kernels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "shopsage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
