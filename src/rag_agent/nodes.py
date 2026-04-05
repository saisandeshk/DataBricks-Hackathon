"""LangGraph node implementations for the agentic legal RAG pipeline."""
from __future__ import annotations

from typing import Literal, Set

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from src.rag_agent.graph_state import AgentState, State
from src.rag_agent.prompts import (
    get_aggregation_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_fallback_response_prompt,
    get_orchestrator_prompt,
    get_rewrite_query_prompt,
)
from src.rag_agent.schemas import QueryAnalysis
from src.config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from src.utils import estimate_context_tokens


# ── Main graph nodes ─────────────────────────────────────────────────────

def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": "", "agent_answers": [{"__reset__": True}]}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]
    if not relevant_msgs:
        return {"conversation_summary": "", "agent_answers": [{"__reset__": True}]}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary = llm.with_config(temperature=0.2).invoke(
        [SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)]
    )
    return {"conversation_summary": summary.content, "agent_answers": [{"__reset__": True}]}


def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    summary = state.get("conversation_summary", "")

    context = (
        (f"Conversation Context:\n{summary}\n" if summary.strip() else "")
        + f"User Query:\n{last_message.content}\n"
    )

    structured_llm = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = structured_llm.invoke(
        [SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context)]
    )

    if response.questions and response.is_clear:
        delete_all = [
            RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
        }

    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "I need more details to understand your legal question. Could you specify the section number, article, or case you're referring to?"
    )
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}


def request_clarification(state: State):
    return {}


# ── Agent subgraph nodes ─────────────────────────────────────────────────

def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary
        else []
    )

    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(
            content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION."
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,
    }


def fallback_response(state: AgentState, llm):
    seen: set[str] = set()
    unique_contents: list[str] = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    parts: list[str] = []
    if context_summary:
        parts.append(f"## Compressed Research Context\n\n{context_summary}")
    if unique_contents:
        parts.append(
            "## Retrieved Data\n\n"
            + "\n\n".join(f"--- SOURCE {i} ---\n{c}" for i, c in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(parts) if parts else "No data was retrieved from the legal documents."

    prompt = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible legal answer using only the data above."
    )
    response = llm.invoke(
        [SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt)]
    )
    return {"messages": [response]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)
                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    token_msgs = estimate_context_tokens(messages)
    token_summary = estimate_context_tokens(
        [HumanMessage(content=state.get("context_summary", ""))]
    )
    total = token_msgs + token_summary
    max_allowed = BASE_TOKEN_THRESHOLD + int(token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if total > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                calls_info = f" | Tool calls: {calls}"
            text += f"[ASSISTANT{calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke(
        [SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=text)]
    )
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parents = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        searches = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))
        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parents:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parents) + "\n"
        if searches:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in searches) + "\n"
        new_summary += block

    return {
        "context_summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [
            {"index": state["question_index"], "question": state["question"], "answer": answer}
        ],
    }


# ── Aggregation (main graph) ─────────────────────────────────────────────

def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted = ""
    for i, ans in enumerate(sorted_answers, 1):
        formatted += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_msg = HumanMessage(
        content=f"Original user question: {state['originalQuery']}\nRetrieved answers:{formatted}"
    )
    synthesis = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_msg])
    return {"messages": [AIMessage(content=synthesis.content)]}
