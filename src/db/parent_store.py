"""Parent chunk store — JSON-file-backed, compatible with agentic-rag pattern.

During ingestion each parent chunk (full BNS section / Constitution article /
SC judgment paragraph) is saved as ``{parent_id}.json``.  At query time the
agent retrieves the full context via ``load_content(parent_id)``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STORE = os.environ.get(
    "PARENT_STORE_DIR",
    "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index",
)
_LOCAL_CACHE = os.environ.get("PARENT_STORE_LOCAL", "/tmp/nyaya_index")


class ParentStoreManager:
    """Persist / retrieve parent chunks as individual JSON files."""

    def __init__(self, store_path: str | None = None):
        self._store_path = Path(store_path or _DEFAULT_STORE)

    # ── helpers ───────────────────────────────────────────────────────────

    def _ensure_local(self) -> Path:
        """If store is on a UC Volume, download to local cache via SDK."""
        from src.db.volume_download import download_volume_dir

        local = Path(_LOCAL_CACHE)
        return download_volume_dir(str(self._store_path), str(local))

    @staticmethod
    def _sort_key(id_str: str) -> int:
        m = re.search(r"_parent_(\d+)$", id_str)
        return int(m.group(1)) if m else 0

    # ── write (notebook) ──────────────────────────────────────────────────

    def save(self, parent_id: str, content: str, metadata: dict[str, Any]) -> None:
        self._store_path.mkdir(parents=True, exist_ok=True)
        fp = self._store_path / f"{parent_id}.json"
        fp.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_many(self, parents: list[tuple[str, Any]]) -> None:
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    # ── read (app) ────────────────────────────────────────────────────────

    def load(self, parent_id: str) -> dict[str, Any]:
        base = self._ensure_local()
        fname = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        fp = base / fname
        return json.loads(fp.read_text(encoding="utf-8"))

    def load_content(self, parent_id: str) -> dict[str, Any]:
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"],
        }

    def load_content_many(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        unique = set(parent_ids)
        results = []
        for pid in sorted(unique, key=self._sort_key):
            try:
                results.append(self.load_content(pid))
            except FileNotFoundError:
                logger.warning("Parent chunk not found: %s", pid)
        return results

    def clear_store(self) -> None:
        if self._store_path.is_dir():
            for child in self._store_path.iterdir():
                child.unlink(missing_ok=True)
