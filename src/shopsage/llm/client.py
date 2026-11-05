"""OpenAI-compatible chat client configured only from .env (works with Ollama, vLLM, OpenAI...)."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from shopsage.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.openai_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout_s)


def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 300,
    model: str | None = None,
) -> str:
    s = get_settings()
    resp = get_client().chat.completions.create(
        model=model or s.llm_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        seed=0,
    )
    return (resp.choices[0].message.content or "").strip()
