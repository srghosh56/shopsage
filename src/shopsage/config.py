"""Paths and settings. Secrets are read from .env only."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SHOPSAGE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = Path(os.environ.get("SHOPSAGE_PROCESSED_DIR", DATA_DIR / "processed"))
ARTIFACTS_DIR = Path(os.environ.get("SHOPSAGE_ARTIFACTS_DIR", ROOT / "artifacts"))
DOCS_DIR = ROOT / "docs"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
DUCKDB_PATH = PROCESSED_DIR / "shopsage.duckdb"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))  # keep downloads in-repo
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Time-based evaluation split (see docs/DECISIONS.md)
TEST_START = date(2011, 9, 1)
TEST_END = date(2011, 12, 1)  # exclusive; three months after TEST_START
VAL_START = date(2011, 6, 1)  # last three months of train are validation
SEED = 42


class Settings(BaseSettings):
    """LLM settings for any OpenAI-compatible endpoint."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-set"
    llm_model: str = "mistral"
    llm_timeout_s: float = 120.0
    # comma-separated judge models for the faithfulness evaluation; empty = llm_model
    judge_models: str = ""
    api_url: str = "http://127.0.0.1:8000"

    @property
    def judges(self) -> list[str]:
        return [m.strip() for m in self.judge_models.split(",") if m.strip()] or [self.llm_model]

    @property
    def openai_base_url(self) -> str:
        url = self.llm_base_url.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
