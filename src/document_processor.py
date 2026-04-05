"""PDF → Markdown → Parent/Child chunks pipeline.

Follows the agentic-rag-for-dummies pattern:
  1. Convert PDF to Markdown via pymupdf4llm
  2. Split Markdown into parent chunks via MarkdownHeaderTextSplitter
  3. Split parents into child chunks via RecursiveCharacterTextSplitter
  4. Return (parent_chunks, child_chunks) for downstream indexing

Usage (in notebooks):
    from src.document_processor import DocumentProcessor
    proc = DocumentProcessor()
    parents, children = proc.process_pdfs("/Volumes/.../pdfs")
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Converts PDFs or Markdown files into parent/child chunk pairs."""

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 100,
        min_parent_size: int = 2000,
        max_parent_size: int = 4000,
        markdown_dir: str | None = None,
    ):
        from src.config import (
            HEADERS_TO_SPLIT_ON,
            CHILD_CHUNK_SIZE,
            CHILD_CHUNK_OVERLAP,
            MIN_PARENT_SIZE,
            MAX_PARENT_SIZE,
            MARKDOWN_DIR,
        )

        self._headers = headers_to_split_on or HEADERS_TO_SPLIT_ON
        self._child_chunk_size = child_chunk_size or CHILD_CHUNK_SIZE
        self._child_chunk_overlap = child_chunk_overlap or CHILD_CHUNK_OVERLAP
        self._min_parent_size = min_parent_size or MIN_PARENT_SIZE
        self._max_parent_size = max_parent_size or MAX_PARENT_SIZE
        self._markdown_dir = Path(markdown_dir or MARKDOWN_DIR)
        self._markdown_dir.mkdir(parents=True, exist_ok=True)

        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers,
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def process_pdfs(
        self, pdf_dir: str, overwrite: bool = False
    ) -> tuple[list[tuple[str, Document]], list[Document]]:
        """Convert all PDFs in *pdf_dir* to Markdown, then chunk.

        Returns (parent_pairs, child_chunks) where parent_pairs is
        ``[(parent_id, Document), ...]`` and child_chunks carry
        ``metadata["parent_id"]``.
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.warning("PDF directory %s does not exist", pdf_dir)
            return [], []

        all_parents: list[tuple[str, Document]] = []
        all_children: list[Document] = []

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            md_path = self._markdown_dir / f"{pdf_path.stem}.md"
            if overwrite or not md_path.exists():
                self._pdf_to_markdown(pdf_path, md_path)
            parents, children = self._chunk_markdown(md_path)
            all_parents.extend(parents)
            all_children.extend(children)
            logger.info(
                "%s → %d parents, %d children",
                pdf_path.name,
                len(parents),
                len(children),
            )

        return all_parents, all_children

    def process_markdown_dir(
        self, md_dir: str | None = None
    ) -> tuple[list[tuple[str, Document]], list[Document]]:
        """Chunk all ``.md`` files in a directory (skip PDF step)."""
        md_dir = Path(md_dir) if md_dir else self._markdown_dir
        all_parents: list[tuple[str, Document]] = []
        all_children: list[Document] = []

        for md_path in sorted(md_dir.glob("*.md")):
            parents, children = self._chunk_markdown(md_path)
            all_parents.extend(parents)
            all_children.extend(children)

        return all_parents, all_children

    def process_single_file(
        self, file_path: str, overwrite: bool = False
    ) -> tuple[list[tuple[str, Document]], list[Document]]:
        """Process one PDF or MD file and return chunks."""
        fp = Path(file_path)
        if fp.suffix.lower() == ".pdf":
            md_path = self._markdown_dir / f"{fp.stem}.md"
            if overwrite or not md_path.exists():
                self._pdf_to_markdown(fp, md_path)
            return self._chunk_markdown(md_path)
        elif fp.suffix.lower() == ".md":
            return self._chunk_markdown(fp)
        else:
            logger.warning("Unsupported file type: %s", fp.suffix)
            return [], []

    # ── PDF → Markdown ────────────────────────────────────────────────────

    @staticmethod
    def _pdf_to_markdown(pdf_path: Path, md_path: Path) -> None:
        """Convert a single PDF to Markdown using pymupdf4llm."""
        import pymupdf
        import pymupdf4llm

        doc = pymupdf.open(str(pdf_path))
        md_text = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None,
        )
        # Clean encoding artifacts
        md_cleaned = (
            md_text.encode("utf-8", errors="surrogatepass")
            .decode("utf-8", errors="ignore")
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_cleaned, encoding="utf-8")
        logger.info("PDF → MD: %s → %s", pdf_path.name, md_path.name)

    # ── Markdown → Parent/Child chunks ────────────────────────────────────

    def _chunk_markdown(
        self, md_path: Path
    ) -> tuple[list[tuple[str, Document]], list[Document]]:
        """Split one Markdown file into parent + child chunks."""
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            return [], []

        raw_parents = self._parent_splitter.split_text(text)
        merged = self._merge_small_parents(raw_parents)
        split = self._split_large_parents(merged)
        cleaned = self._clean_small_chunks(split)

        parent_pairs: list[tuple[str, Document]] = []
        child_chunks: list[Document] = []

        for i, p_chunk in enumerate(cleaned):
            parent_id = f"{md_path.stem}_parent_{i}"
            p_chunk.metadata.update({
                "source": f"{md_path.stem}.pdf",
                "parent_id": parent_id,
            })
            parent_pairs.append((parent_id, p_chunk))
            children = self._child_splitter.split_documents([p_chunk])
            child_chunks.extend(children)

        return parent_pairs, child_chunks

    # ── Merge / split helpers (from agentic-rag-for-dummies) ──────────────

    def _merge_small_parents(self, chunks: list[Document]) -> list[Document]:
        if not chunks:
            return []
        merged: list[Document] = []
        current: Document | None = None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self._min_parent_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)

        return merged

    def _split_large_parents(self, chunks: list[Document]) -> list[Document]:
        result: list[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) <= self._max_parent_size:
                result.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._max_parent_size,
                    chunk_overlap=self._child_chunk_overlap,
                )
                result.extend(splitter.split_documents([chunk]))
        return result

    def _clean_small_chunks(self, chunks: list[Document]) -> list[Document]:
        cleaned: list[Document] = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self._min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = (
                        chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    )
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = (
                                f"{v} -> {chunks[i + 1].metadata[k]}"
                            )
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        return cleaned
