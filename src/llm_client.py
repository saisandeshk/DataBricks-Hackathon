"""Databricks-compatible LLM client — wraps AI Gateway / PAT / OAuth M2M.

Provides both raw `chat_completions()` for the old pipeline AND a
LangChain-compatible `ChatDatabricks` wrapper for LangGraph nodes.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Low-level OpenAI-compatible HTTP helpers (carried from hackathon)
# ---------------------------------------------------------------------------

def _chat_url() -> str:
    full = os.environ.get("LLM_CHAT_COMPLETIONS_URL", "").strip()
    if full:
        return full
    base = os.environ.get("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "Set LLM_CHAT_COMPLETIONS_URL or LLM_OPENAI_BASE_URL."
        )
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _bearer() -> str:
    token = (
        os.environ.get("DATABRICKS_TOKEN", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if token:
        return token
    return _sdk_oauth_token()


def _sdk_oauth_token() -> str:
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        if w.config.token:
            return w.config.token
        result = w.config.authenticate()
        if isinstance(result, dict):
            auth = result.get("Authorization", "")
            if auth.startswith("Bearer "):
                return auth[7:]
        if callable(result):
            headers = result()
            if isinstance(headers, dict):
                auth = headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    return auth[7:]
    except Exception as exc:
        logger.warning("SDK OAuth token failed: %s", exc)
    return ""


def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST OpenAI-compatible chat/completions; returns parsed JSON body."""
    url = _chat_url()
    token = _bearer()
    if not token:
        raise RuntimeError("Set DATABRICKS_TOKEN, LLM_API_KEY, or OPENAI_API_KEY.")
    model = (model or os.environ.get("LLM_MODEL", "")).strip()
    if not model:
        raise RuntimeError("Set LLM_MODEL.")
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def extract_assistant_text(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected LLM response shape: {response!r}") from e


# ---------------------------------------------------------------------------
# LangChain ChatOpenAI wrapper pointing at Databricks AI Gateway
# ---------------------------------------------------------------------------

def get_langchain_llm(temperature: float | None = None, max_tokens: int = 3072):
    """Return a LangChain ChatOpenAI configured for the Databricks AI Gateway.

    Works with tool_calling, structured output, and streaming.
    """
    from langchain_openai import ChatOpenAI

    base = os.environ.get("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "databricks-llama-4-maverick").strip()
    token = _bearer()
    if not token:
        raise RuntimeError("Set DATABRICKS_TOKEN, LLM_API_KEY, or OPENAI_API_KEY.")
    if not base:
        raise RuntimeError("Set LLM_OPENAI_BASE_URL.")

    return ChatOpenAI(
        model=model,
        openai_api_base=base,
        openai_api_key=token,
        temperature=temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
    )
