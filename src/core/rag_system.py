"""RAGSystem — initialises LLM, FAISS, parent store, and LangGraph agent."""
from __future__ import annotations

import uuid

from src.config import GRAPH_RECURSION_LIMIT
from src.db.faiss_manager import FaissManager
from src.db.parent_store import ParentStoreManager
from src.llm_client import get_langchain_llm
from src.rag_agent.graph import create_agent_graph
from src.rag_agent.tools import ToolFactory


class RAGSystem:
    """High-level container wiring all components together."""

    def __init__(self):
        self.faiss = FaissManager()
        self.parent_store = ParentStoreManager()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = GRAPH_RECURSION_LIMIT

    def initialize(self):
        """Load FAISS index, build tools, compile the LangGraph agent."""
        self.faiss.load()

        llm = get_langchain_llm()
        tools = ToolFactory(self.faiss, self.parent_store).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
