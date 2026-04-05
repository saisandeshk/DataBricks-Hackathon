"""Streaming chat interface — bridges LangGraph agent ↔ Gradio chatbot."""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

SILENT_NODES = {"rewrite_query"}
SYSTEM_NODES = {"summarize_history", "rewrite_query"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query": {"title": "🔍 Query Analysis & Rewriting"},
    "summarize_history": {"title": "📋 Chat History Summary"},
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def _find_msg_idx(messages, node):
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def _parse_rewrite_json(buf):
    m = re.search(r"\{.*\}", buf, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _format_rewrite_content(buf):
    data = _parse_rewrite_json(buf)
    if not data:
        return "⏳ Analyzing query..."
    if data.get("is_clear"):
        lines = ["✅ **Query is clear**"]
        if data.get("questions"):
            lines += ["\n**Rewritten queries:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **Query is unclear**"]
        c = data.get("clarification_needed", "")
        if c and c.strip().lower() != "no":
            lines.append(f"\nClarification needed: *{c}*")
    return "\n".join(lines)


# ── ChatInterface ────────────────────────────────────────────────────────

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def _handle_system_node(self, chunk, node, msgs, buf):
        buf[node] = buf.get(node, "") + chunk.content
        title = SYSTEM_NODE_CONFIG[node]["title"]
        content = _format_rewrite_content(buf[node]) if node == "rewrite_query" else buf[node]

        idx = _find_msg_idx(msgs, node)
        if idx is None:
            msgs.append(_make_message(content, title=title, node=node))
        else:
            msgs[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buf[node], msgs)

    def _surface_clarification(self, buf, msgs):
        data = _parse_rewrite_json(buf) or {}
        c = data.get("clarification_needed", "")
        if not data.get("is_clear") and c.strip().lower() not in ("", "no"):
            idx = _find_msg_idx(msgs, "clarification")
            if idx is None:
                msgs.append(_make_message(c, node="clarification"))
            else:
                msgs[idx]["content"] = c

    def _handle_tool_call(self, chunk, msgs, active):
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active:
                msgs.append(_make_message(f"Running `{tc['name']}`...", title=f"🛠️ {tc['name']}"))
                active[tc["id"]] = len(msgs) - 1

    def _handle_tool_result(self, chunk, msgs, active):
        idx = active.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix = "\n..." if len(str(chunk.content)) > 300 else ""
            msgs[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, msgs):
        last = msgs[-1] if msgs else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            msgs.append(_make_message(""))
        msgs[-1]["content"] += chunk.content

    def chat(self, message: str, history):
        """Generator yielding streaming Gradio chat message dicts."""
        if not self.rag_system.agent_graph:
            yield "⚠️ System not initialized!"
            return

        config = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)

        try:
            if current_state.next:
                self.rag_system.agent_graph.update_state(
                    config, {"messages": [HumanMessage(content=message.strip())]}
                )
                stream_input = None
            else:
                stream_input = {"messages": [HumanMessage(content=message.strip())]}

            response_messages: list[dict] = []
            active_tool_calls: dict[str, int] = {}
            system_node_buffer: dict[str, str] = {}

            for chunk, metadata in self.rag_system.agent_graph.stream(
                stream_input, config=config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")

                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)
                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)
                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)
                elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                    self._handle_llm_token(chunk, response_messages)

                yield response_messages

        except Exception as e:
            yield f"❌ Error: {e!s}"

    def clear_session(self):
        self.rag_system.reset_thread()
