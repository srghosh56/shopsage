export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.python
export TOKENIZERS_PARALLELISM := false
# judges for `make explain-eval` (any models your LLM endpoint serves); empty = LLM_MODEL
JUDGE_MODELS ?= mistral,gemma3:latest
API_PORT ?= 8000
UI_PORT ?= 8501
RUN := uv run
PY := cpython-3.12-macos-aarch64-none

.PHONY: setup data eda train eval clv serve ui mcp agent-eval explain-eval faithfulness test lint all

setup:  ## create the venv with pinned dependencies
	uv python install $(PY) || uv python install 3.12
	uv sync --frozen

data:  ## download Online Retail II, clean it with DuckDB, write Parquet
	$(RUN) python -m shopsage.data.download
	$(RUN) python -m shopsage.data.clean

eda:  ## execute the EDA notebook in place
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb

train:  ## tune item-kNN and ALS on validation (logged to MLflow), fit final models, segments + CLV
	$(RUN) python -m shopsage.recsys.train
	$(RUN) python -m shopsage.analytics.segments
	$(RUN) python -m shopsage.analytics.clv

eval:  ## evaluate every model on the test window and write docs/RESULTS.md tables
	$(RUN) python -m shopsage.recsys.evaluate

serve:  ## FastAPI on :$(API_PORT)
	$(RUN) uvicorn shopsage.api.main:app --host 0.0.0.0 --port $(API_PORT)

ui:  ## Streamlit on :8501 (needs `make serve`)
	API_URL=http://127.0.0.1:$(API_PORT) $(RUN) streamlit run app/streamlit_app.py --server.port $(UI_PORT)

mcp:  ## MCP server over stdio
	$(RUN) python -m shopsage.agent.mcp_server

explain-eval:  ## LLM judge on 30 explanations + write docs/labels/explanations.csv
	JUDGE_MODELS="$(JUDGE_MODELS)" $(RUN) python -m shopsage.llm.judge

agent-eval:  ## 40-question assistant evaluation vs the same LLM without tools
	$(RUN) python -m shopsage.agent.evaluate

faithfulness:  ## agreement between human labels and the LLM judges
	$(RUN) python -m shopsage.llm.agreement

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src app tests
	$(RUN) ruff format --check src app tests
	$(RUN) mypy src app

all: data eda train eval explain-eval agent-eval
