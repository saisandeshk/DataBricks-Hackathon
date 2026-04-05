"""Nyaya Sahayak entry point — Databricks Apps compatible."""
from __future__ import annotations

import os
import sys

# Load .env file if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# Ensure project root is on sys.path so `src.*` imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.ui.gradio_app import create_gradio_ui  # noqa: E402


def main():
    demo = create_gradio_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "8000")),
        share=False,
    )


if __name__ == "__main__":
    main()
