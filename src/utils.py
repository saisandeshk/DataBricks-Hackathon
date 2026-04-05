"""Shared utility functions."""
from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def estimate_context_tokens(messages: list) -> int:
    """Rough token count for a list of LangChain messages."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return sum(
            len(encoding.encode(str(msg.content)))
            for msg in messages
            if hasattr(msg, "content") and msg.content
        )
    except ImportError:
        # Fallback: ~4 chars per token
        return sum(
            len(str(msg.content)) // 4
            for msg in messages
            if hasattr(msg, "content") and msg.content
        )
