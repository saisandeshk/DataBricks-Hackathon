"""LangChain tools for the agentic RAG pipeline — FAISS backed."""
from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from src.db.parent_store import ParentStoreManager


class ToolFactory:
    """Creates search + retrieval tools bound to a FAISS collection."""

    def __init__(self, collection, parent_store: ParentStoreManager | None = None):
        self.collection = collection  # FaissManager or LangChain FAISS store
        self.parent_store = parent_store or ParentStoreManager()

    def _search_child_chunks(self, query: str, limit: int = 8) -> str:
        """Search for the most relevant child chunks in the legal knowledge base.

        Args:
            query: Search query string about Indian law (BNS, Constitution, SC judgments)
            limit: Maximum number of results to return
        """
        try:
            results = self.collection.similarity_search(query, k=limit)
            if not results:
                return "NO_RELEVANT_CHUNKS"
            return "\n\n".join(
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"Source: {doc.metadata.get('source', '')}\n"
                f"Type: {doc.metadata.get('doc_type', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            )
        except Exception as e:
            return f"RETRIEVAL_ERROR: {e!s}"

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunk by its ID for complete legal context.

        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"
            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"Source: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Type: {parent.get('metadata', {}).get('doc_type', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
        except FileNotFoundError:
            return f"PARENT_NOT_FOUND: {parent_id}"
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {e!s}"

    def create_tools(self) -> List:
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        return [search_tool, retrieve_tool]
