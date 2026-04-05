"""Sentence embeddings for RAG (same model at index build and query time)."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Thin wrapper around sentence-transformers with lazy loading."""

    def __init__(self, model_name: str = DEFAULT_MODEL, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return int(self._load_model().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> "NDArray[np.float32]":
        m = self._load_model()
        emb = m.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(emb, dtype=np.float32)


@lru_cache(maxsize=4)
def get_embedder(model_name: str = DEFAULT_MODEL, normalize: bool = True) -> SentenceEmbedder:
    return SentenceEmbedder(model_name=model_name, normalize=normalize)
