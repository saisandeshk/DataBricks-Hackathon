"""FAISS vector store manager — LangChain-compatible drop-in for Qdrant.

Provides ``similarity_search`` returning LangChain ``Document`` objects so the
``ToolFactory`` from the agentic-rag pipeline works unchanged.

Index lifecycle:
  * **Build** (notebook): call ``build_and_save(texts, metadatas)``
  * **Query** (app):      call ``load()`` then ``similarity_search()``
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default paths — overridden by env / config at runtime
_DEFAULT_INDEX_DIR = os.environ.get(
    "FAISS_INDEX_DIR",
    "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index",
)
_LOCAL_CACHE = os.environ.get("FAISS_LOCAL_CACHE", "/tmp/nyaya_index")


class FaissManager:
    """Manages a LangChain FAISS vector store backed by sentence-transformers."""

    def __init__(
        self,
        index_dir: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._index_dir = index_dir or _DEFAULT_INDEX_DIR
        self._embedding_model = embedding_model
        self._store = None
        self._embeddings = None

    # ── Lazy embedding function ──────────────────────────────────────────

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    # ── Build (used in notebooks) ────────────────────────────────────────

    def build_and_save(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        save_dir: str | None = None,
    ) -> None:
        """Create a FAISS index from scratch and persist to disk."""
        from langchain_community.vectorstores import FAISS

        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        logger.info("Building FAISS index from %d documents …", len(docs))
        self._store = FAISS.from_documents(docs, self._get_embeddings())
        dest = save_dir or self._index_dir
        Path(dest).mkdir(parents=True, exist_ok=True)
        self._store.save_local(dest)
        logger.info("FAISS index saved to %s (%d vectors)", dest, len(texts))

    # ── Load (used at app start) ─────────────────────────────────────────

    def load(self, index_dir: str | None = None) -> None:
        """Load a persisted FAISS index."""
        from langchain_community.vectorstores import FAISS
        from src.db.volume_download import download_volume_dir

        src = index_dir or self._index_dir
        # Download from UC Volume if direct path isn't available
        local = Path(_LOCAL_CACHE)
        load_path = str(download_volume_dir(src, str(local)))

        self._store = FAISS.load_local(
            load_path, self._get_embeddings(), allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", load_path)

    # ── Query ────────────────────────────────────────────────────────────

    @property
    def store(self):
        if self._store is None:
            self.load()
        return self._store

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """Return top-k Documents with metadata (compatible with ToolFactory)."""
        if score_threshold is not None:
            retriever = self.store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": k},
            )
            return retriever.invoke(query)
        return self.store.similarity_search(query, k=k)

    def as_langchain_store(self):
        """Return the underlying LangChain FAISS object."""
        return self.store
