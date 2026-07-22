import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import html
import re
import streamlit as st
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

from eval.dashboard import prepare_trend_data


# ── Reusable API helpers with optional JWT auth ────────────────────────


def _get_auth_headers() -> Dict[str, str]:
    """Return Authorization header dict if a JWT token is stored in session state."""
    token = st.session_state.get("jwt_token", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def api_get(api_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> requests.Response:
    """Reusable GET helper with optional JWT auth."""
    headers = _get_auth_headers()
    url = f"{api_url}{endpoint}"
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def api_post(api_url: str, endpoint: str, json: Optional[Dict[str, Any]] = None, timeout: int = 30) -> requests.Response:
    """Reusable POST helper with optional JWT auth."""
    headers = _get_auth_headers()
    url = f"{api_url}{endpoint}"
    return requests.post(url, headers=headers, json=json, timeout=timeout)


def api_delete(api_url: str, endpoint: str, timeout: int = 10) -> requests.Response:
    """Reusable DELETE helper with optional JWT auth."""
    headers = _get_auth_headers()
    url = f"{api_url}{endpoint}"
    return requests.delete(url, headers=headers, timeout=timeout)


def inject_chat_style() -> None:
    st.markdown(
        """
        <style>
        .chat-area { display: flex; flex-direction: column; gap: 10px; width: 100%; }
        .chat-container { width: 100%; }
        .chat-message { display: flex; width: 100%; margin: 6px 0; }
        .chat-message--user { justify-content: flex-end; }
        .chat-message--assistant { justify-content: flex-start; }
        .chat-bubble {
            max-width: 70%;
            min-width: 28%;
            padding: 16px 20px;
            border-radius: 22px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .chat-message--user .chat-bubble {
            background: #0d6efd;
            color: white;
            border-bottom-right-radius: 10px;
            border-bottom-left-radius: 22px;
            border-top-right-radius: 22px;
            border-top-left-radius: 22px;
        }
        .chat-message--assistant .chat-bubble {
            background: #ffffff;
            color: #111;
            border: 1px solid #e9edf2;
            border-bottom-left-radius: 10px;
            border-bottom-right-radius: 22px;
            border-top-left-radius: 22px;
            border-top-right-radius: 22px;
        }
        .chat-bubble-title { display: block; margin-bottom: 8px; font-size: 0.95rem; font-weight: 700; opacity: 0.85; }
        .chat-bubble-text { font-size: 1rem; }
        .chat-bubble-text a { color: #0d6efd; text-decoration: underline; font-weight: 500; }
        .chat-bubble-text a:hover { text-decoration: none; }
        .citation-block { margin-top: 8px; padding: 12px 14px; background: #f9fafc; border: 1px solid #e7ecf2; border-radius: 16px; max-width: 72%; }
        .chat-message--user .citation-block { margin-left: auto; }
        .citation-header { font-weight: 700; margin-bottom: 8px; }
        .citation-item { margin-bottom: 8px; font-size: 0.95rem; }
        .citation-item a { color: #0d6efd; text-decoration: none; }
        .citation-item a:hover { text-decoration: underline; }
        .chat-details { margin-top: 8px; font-size: 0.9rem; color: #555; }
        .chat-details summary { cursor: pointer; font-weight: 600; }
        .chat-details p { margin: 4px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_chunk_id_to_source_map(citations: list) -> Dict[str, str]:
    """Build a mapping from chunk_id to source_id from the citations list."""
    mapping: Dict[str, str] = {}
    for citation in citations:
        chunk_id = citation.get("chunk_id", "")
        source_id = citation.get("source_id", "")
        if chunk_id and source_id:
            mapping[chunk_id] = source_id
    return mapping


def _make_chunk_id_links(text: str, chunk_to_source: Dict[str, str], api_url: str) -> str:
    """Replace [chunk_id] references in text with clickable HTML links to the source download endpoint.
    
    Args:
        text: The answer text containing [chunk_id] references.
        chunk_to_source: Mapping from chunk_id to source_id.
        api_url: Backend API URL for generating download links.
    
    Returns:
        HTML string with [chunk_id] references replaced by anchor tags.
    """
    def _replace_match(match: re.Match) -> str:
        chunk_id = match.group(1)
        source_id = chunk_to_source.get(chunk_id)
        if source_id and api_url:
            download_url = f"{api_url}/sources/{source_id}/download"
            return f'<a href="{download_url}" target="_blank" rel="noopener noreferrer" title="View source document: {chunk_id}">[{chunk_id}]</a>'
        # If no mapping found, keep the original text but style it
        return f'<span class="citation-ref">[{chunk_id}]</span>'
    
    # Pattern matches [chunk_id] where chunk_id is alphanumeric with underscores/hyphens
    pattern = r'\[([a-zA-Z0-9_\-]+)\]'
    return re.sub(pattern, _replace_match, text)


def render_chat_bubble(role: str, text: str, metadata: Optional[Dict[str, Any]] = None, index: int = 0,
                       citations: Optional[list] = None, api_url: str = "") -> None:
    """Render a chat bubble with optional clickable [chunk_id] citation links.
    
    Args:
        role: 'user' or 'assistant'
        text: The message text.
        metadata: Optional metadata dict (unused, kept for backward compatibility).
        index: Message index for unique key generation.
        citations: List of citation dicts for building chunk_id → source_id mapping.
        api_url: Backend API URL for generating download links.
    """
    role_class = "user" if role == "user" else "assistant"
    title = "You" if role == "user" else "Assistant"
    
    if role == "assistant" and citations:
        # Build chunk_id → source_id mapping and convert [chunk_id] references to links
        chunk_to_source = _build_chunk_id_to_source_map(citations)
        display_text = _make_chunk_id_links(text, chunk_to_source, api_url)
    else:
        display_text = html.escape(text).replace("\n", "<br />")
    
    st.markdown(
        f"""
        <div class='chat-container'>
            <div class='chat-message chat-message--{role_class}'>
                <div class='chat-bubble'>
                    <span class='chat-bubble-title'>{title}</span>
                    <span class='chat-bubble-text'>{display_text}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # JSON context/details are now shown in the sidebar; do not display per-message metadata here.
    return


def render_citations(citations: list, message_index: str = "0", api_url: str = "") -> None:
    """Render clickable source citations below the chat bubble with expandable chunk details.
    
    Args:
        citations: List of citation dictionaries
        message_index: Index of the message in conversation history for unique key generation (can include prefix like 'live_')
        api_url: Backend API URL for generating download links
    """
    if not citations:
        return

    st.markdown("<div class='citation-block'><div class='citation-header'>📚 Sources & chunks</div>", unsafe_allow_html=True)

    for idx, citation in enumerate(citations, 1):
        source_id = citation.get("source_id", "unknown")
        source_path = citation.get("source_path", "")
        doc_type = citation.get("doc_type", "document")
        score = citation.get("score", 0.0)
        chunk_id = citation.get("chunk_id", "N/A")
        chunk_text = citation.get("text", "")

        # Build a download link to the original document
        download_url = f"{api_url}/sources/{source_id}/download" if api_url else ""

        if source_path and (source_path.startswith("http://") or source_path.startswith("https://")):
            source_line = f"**[{idx}]** [{source_id}]({source_path}) - *{doc_type}* (score: {score:.2f})"
        elif download_url:
            source_line = f"**[{idx}]** [{source_id}]({download_url}) - `{source_path}` - *{doc_type}* (score: {score:.2f})"
        elif source_path:
            source_line = f"**[{idx}]** **{source_id}** - `{source_path}` - *{doc_type}* (score: {score:.2f})"
        else:
            source_line = f"**[{idx}]** **{source_id}** - *{doc_type}* (score: {score:.2f})"

        st.markdown(f"<div class='citation-item'>{source_line}</div>", unsafe_allow_html=True)

        with st.expander(f"📄 View Chunk Details [{idx}]", expanded=False):
            st.markdown(f"**Chunk ID:** `{chunk_id}`")
            st.markdown(f"**Source ID:** {source_id}")
            st.markdown(f"**Relevance Score:** {score:.4f}")
            if chunk_text:
                st.markdown("**Chunk Text:**")
                st.text_area(
                    "Chunk content",
                    value=chunk_text,
                    height=120,
                    key=f"chunk_text_{message_index}_cit{idx}_{chunk_id}",
                    label_visibility="collapsed",
                )
    st.markdown("</div>", unsafe_allow_html=True)


def get_backend_health(api_url: str) -> Optional[Dict[str, Any]]:
    try:
        response = api_get(api_url, "/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None


def get_saved_history(api_url: str, session_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = api_get(api_url, f"/history/{session_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None


def call_chat_api(session_id: str, message: str, api_url: str) -> Dict[str, Any]:
    payload = {
        "session_id": session_id,
        "message": message,
    }
    response = api_post(api_url, "/chat", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Claims Processing & Settlement", page_icon="🛡️", layout="wide",
                       initial_sidebar_state="expanded")
    st.title("Claims Processing & Settlement")
    inject_chat_style()

    query_params = st.query_params
    if "session_id" not in st.session_state:
        st.session_state.session_id = query_params.get("session_id", [None])[0] or uuid4().hex
        st.query_params = {"session_id": st.session_state.session_id}
    else:
        st.query_params = {"session_id": st.session_state.session_id}

    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_langsmith_trace_id" not in st.session_state:
        st.session_state.latest_langsmith_trace_id = None

    api_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

    st.sidebar.header("Session & Backend")
    st.sidebar.text_input("Session ID", value=st.session_state.session_id, disabled=True)

    health = get_backend_health(api_url)
    if health:
        st.sidebar.write(f"**Model:** {health.get('model', 'unknown')}")
        st.sidebar.write(f"**Temperature:** {health.get('temperature', 'unknown')}")
        st.sidebar.write(f"**DB status:** {health.get('db_status', 'unknown')}")
        st.sidebar.write(f"**Uptime:** {health.get('uptime_seconds', '?')}s")
    else:
        st.sidebar.warning(f"Backend health unavailable for {api_url}.")

    st.sidebar.divider()
    st.sidebar.header("LangSmith Tracing")
    if st.session_state.latest_langsmith_trace_id:
        st.sidebar.success(f"✓ Tracing enabled")
        st.sidebar.code(st.session_state.latest_langsmith_trace_id, language="text")
        st.sidebar.markdown(f"🔗 [View in LangSmith](https://smith.langchain.com)", unsafe_allow_html=True)
    else:
        st.sidebar.info("No trace ID yet. Send a message to generate a trace.")

    # ── Role Selector & JWT Auth ────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.header("🔑 Role & Auth")

    # Available roles matching config/roles.yaml
    ROLE_OPTIONS = [
        "claims_processor",
        "senior_adjuster",
        "claims_manager",
        "fraud_investigator",
    ]

    selected_role = st.sidebar.selectbox(
        "Select Role",
        options=ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(st.session_state.get("selected_role", "claims_processor")),
        key="role_selector",
    )

    col_token_btn, col_clear_btn = st.sidebar.columns(2)
    with col_token_btn:
        if st.button("🔄 Get Token", use_container_width=True):
            try:
                token_resp = api_post(
                    api_url, "/auth/token",
                    json={"sub": "demo_user", "role": selected_role},
                    timeout=10,
                )
                if token_resp.status_code == 200:
                    data = token_resp.json()
                    st.session_state.jwt_token = data.get("access_token", "")
                    st.session_state.selected_role = selected_role
                    st.rerun()
                else:
                    st.sidebar.error(f"Token request failed: {token_resp.status_code}")
            except requests.RequestException as e:
                st.sidebar.error(f"Error: {e}")

    with col_clear_btn:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.jwt_token = ""
            st.session_state.pop("auth_context", None)
            st.rerun()

    # Show token status
    if st.session_state.get("jwt_token"):
        st.sidebar.success(f"✓ Token active ({st.session_state.get('selected_role', '?')})")
    else:
        st.sidebar.info("No token — anonymous access")

    # ── Auth Context Panel ──────────────────────────────────────────────
    if st.session_state.get("jwt_token"):
        with st.sidebar.expander("Auth Context", expanded=False):
            try:
                ctx_resp = api_get(api_url, "/auth/context", timeout=10)
                if ctx_resp.status_code == 200:
                    ctx = ctx_resp.json()
                    st.session_state.auth_context = ctx
                    st.json(ctx)
                else:
                    st.error(f"Failed: {ctx_resp.status_code}")
            except requests.RequestException as e:
                st.error(f"Error: {e}")
    else:
        # Show anonymous context when no token
        with st.sidebar.expander("Auth Context (anonymous)", expanded=False):
            try:
                ctx_resp = api_get(api_url, "/auth/context", timeout=10)
                if ctx_resp.status_code == 200:
                    st.json(ctx_resp.json())
            except requests.RequestException:
                st.info("Backend unreachable")

    # Response context panel: show JSON context for the latest response in the left panel
    st.sidebar.divider()
    st.sidebar.header("Latest Response Context")
    if st.session_state.history:
        last = st.session_state.history[-1]
        answer_text = last.get("answer_text", "")
        structured = last.get("structured", {}) or {}
        chain_metadata = last.get("chain_metadata", {}) or {}
        if answer_text:
            st.sidebar.write(f"**Answer:** {answer_text}")
            with st.sidebar.expander("JSON Context", expanded=False):
                ctx = {
                    "intent": structured.get("intent"),
                    "category": structured.get("category"),
                    "confidence": structured.get("confidence"),
                    "model": chain_metadata.get("model"),
                    "temperature": chain_metadata.get("temperature"),
                    "reasoning": structured.get("reasoning"),
                    "langsmith_trace_id": chain_metadata.get("langsmith_trace_id"),
                    "extra_metadata": structured.get("metadata"),
                }
                st.json(ctx)
        else:
            st.sidebar.write("No response yet.")

    history_validation = st.sidebar.empty()
    if st.sidebar.button("Validate saved history"):
        saved = get_saved_history(api_url, st.session_state.session_id)
        if saved is not None:
            history_validation.success(
                f"Saved messages: {saved.get('message_count', 0)} | Turns: {saved.get('turn_count', 0)}"
            )
            history_validation.write(saved.get("history", []))
        else:
            history_validation.error("Unable to fetch saved history from backend.")

    # ── Main area tabs: Chat | HITL Review | Prompt Versions | Evaluation Dashboard ──
    tab_chat, tab_hitl, tab_prompts, tab_eval = st.tabs(
        ["💬 Chat", "🛂 HITL Review", "📝 Prompt Versions", "📊 Evaluation Dashboard"]
    )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Chat (existing behavior, unchanged)
    # ═══════════════════════════════════════════════════════════════════
    with tab_chat:
        if st.session_state.history:
            st.markdown("---")
            st.subheader("Conversation History")
            st.write(f"**Local turns:** {len(st.session_state.history)}")
            for idx, item in enumerate(st.session_state.history):
                user_msg = item["user"]
                answer_text = item["answer_text"]
                structured = item.get("structured", {})
                chain_metadata = item.get("chain_metadata", {})
                citations = item.get("citations", [])

                render_chat_bubble("user", user_msg)
                render_chat_bubble("assistant", answer_text, None, idx, citations=citations, api_url=api_url)
                if citations:
                    render_citations(citations, str(idx), api_url)
                with st.expander("Response metadata", expanded=False):
                    st.write("**Structured response**")
                    st.json(structured)
                    st.write("**Chain metadata**")
                    st.json(chain_metadata)

        if hasattr(st, "chat_input"):
            query = st.chat_input("Ask a question about claims, coverage, or settlement")
        else:
            query = st.text_input("Ask a question about claims, coverage, or settlement", key="chat_query_input")

        if query:
            try:
                resp = call_chat_api(st.session_state.session_id, query, api_url)
                structured = resp.get("structured", {})
                chain_metadata = resp.get("chain_metadata", {})
                citations = resp.get("citations", [])
                retrieval_trace = resp.get("retrieval_trace", [])
                st.session_state.latest_langsmith_trace_id = chain_metadata.get("langsmith_trace_id")
                st.session_state.history.append(
                    {
                        "user": query,
                        "answer_text": resp.get("answer_text", ""),
                        "structured": structured,
                        "chain_metadata": chain_metadata,
                        "citations": citations,
                        "retrieval_trace": retrieval_trace,
                    }
                )
                st.rerun()
            except requests.RequestException as exc:
                st.warning(f"Unable to reach backend at {api_url}: {exc}")
                st.session_state.history.append(
                    {
                        "user": query,
                        "answer_text": (
                            f"Sorry, the backend at {api_url} is unavailable. "
                            "Please verify the backend URL and ensure the backend is running."
                        ),
                        "structured": {},
                        "chain_metadata": {"error": str(exc)},
                        "citations": [],
                        "retrieval_trace": [],
                    }
                )
                st.rerun()

        if st.button("Reset Conversation"):
            try:
                api_post(
                    api_url, "/reset",
                    json={"session_id": st.session_state.session_id},
                    timeout=10,
                )
            except requests.RequestException:
                st.warning("Could not reset the session on the backend.")
            st.session_state.history = []
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: HITL Review
    # ═══════════════════════════════════════════════════════════════════
    with tab_hitl:
        st.subheader("🛂 Human-In-The-Loop Review")
        st.caption("Review pending actions that require human approval before execution.")

        # Refresh button
        col_refresh, col_status = st.columns([1, 4])
        with col_refresh:
            refresh_clicked = st.button("🔄 Refresh", key="hitl_refresh")
        with col_status:
            hitl_enabled_placeholder = st.empty()

        # Fetch pending tasks
        try:
            pending_resp = api_get(api_url, "/hitl/pending", timeout=10)
            if pending_resp.status_code == 200:
                pending_data = pending_resp.json()
                tasks = pending_data.get("tasks", [])
                hitl_enabled = pending_data.get("enabled", False)
                task_count = pending_data.get("count", 0)

                if not hitl_enabled:
                    hitl_enabled_placeholder.warning(
                        "⚠️ HITL is disabled on the backend. Set ENABLE_HITL=true to enable."
                    )
                elif task_count == 0:
                    hitl_enabled_placeholder.info("✅ No pending HITL tasks.")
                else:
                    hitl_enabled_placeholder.success(f"{task_count} pending task(s) awaiting review.")
            else:
                hitl_enabled_placeholder.error(f"Failed to fetch tasks: {pending_resp.status_code}")
                tasks = []
        except requests.RequestException as e:
            hitl_enabled_placeholder.error(f"Error connecting to backend: {e}")
            tasks = []

        # Render each pending task
        for task in tasks:
            task_id = task.get("task_id", "unknown")
            session_id = task.get("session_id", "")
            user_message = task.get("user_message", "")
            agent_response = task.get("agent_response", "")
            rule_id = task.get("rule_id", "")
            rule_reason = task.get("rule_reason", "")
            retrieved_chunks = task.get("retrieved_chunks", [])
            reasoning_trace = task.get("reasoning_trace", "")
            confidence = task.get("confidence", 0.0)
            recommendation = task.get("recommendation", {})
            created_at = task.get("created_at", "")

            with st.container(border=True):
                st.markdown(f"**Task:** `{task_id}`")
                st.caption(f"Session: `{session_id}` | Created: {created_at}")

                # Rule info
                st.markdown(f"**Trigger Rule:** `{rule_id}` — {rule_reason}")

                # User question / request
                if user_message:
                    with st.expander("💬 User Message", expanded=True):
                        st.write(user_message)

                # Agent response (proposed)
                if agent_response:
                    with st.expander("🤖 Proposed Agent Response", expanded=True):
                        st.write(agent_response)

                # Retrieved chunks
                if retrieved_chunks:
                    with st.expander(f"📚 Retrieved Chunks ({len(retrieved_chunks)})", expanded=False):
                        for ci, chunk in enumerate(retrieved_chunks, 1):
                            chunk_text = chunk.get("text", chunk.get("page_content", ""))
                            source_id = chunk.get("source_id", chunk.get("metadata", {}).get("source_id", "unknown"))
                            score = chunk.get("score", chunk.get("metadata", {}).get("score", 0.0))
                            st.markdown(f"**Chunk {ci}** — Source: `{source_id}` (score: {score:.3f})")
                            st.text_area(
                                f"Chunk {ci} content",
                                value=chunk_text,
                                height=120,
                                key=f"hitl_chunk_{task_id}_{ci}",
                                label_visibility="collapsed",
                            )
                            st.divider()

                # Reasoning trace
                if reasoning_trace:
                    with st.expander("🧠 Reasoning Trace", expanded=False):
                        st.text_area(
                            "Reasoning",
                            value=reasoning_trace,
                            height=150,
                            key=f"hitl_reasoning_{task_id}",
                            label_visibility="collapsed",
                        )

                # Confidence & proposed action
                col_conf, col_action = st.columns(2)
                with col_conf:
                    st.metric("Confidence", f"{confidence:.2%}" if confidence else "N/A")
                with col_action:
                    action_type = recommendation.get("action", recommendation.get("type", "unknown"))
                    st.metric("Proposed Action", action_type)

                # Recommendation details
                if recommendation:
                    with st.expander("📋 Recommendation Details", expanded=False):
                        st.json(recommendation)

                # ── Review controls ──────────────────────────────────────
                st.markdown("**Review Decision**")
                review_comments = st.text_area(
                    "Reviewer Comments (optional)",
                    placeholder="Add comments explaining your decision...",
                    key=f"hitl_comments_{task_id}",
                )

                col_approve, col_reject, col_msg = st.columns([1, 1, 3])
                with col_approve:
                    if st.button("✅ Approve", key=f"hitl_approve_{task_id}", type="primary", use_container_width=True):
                        try:
                            review_resp = api_post(
                                api_url, f"/hitl/review/{task_id}",
                                json={"decision": "approved", "comments": review_comments},
                                timeout=10,
                            )
                            if review_resp.status_code == 200:
                                st.success(f"✅ Task `{task_id}` approved!")
                                st.rerun()
                            else:
                                st.error(f"Failed: {review_resp.status_code} - {review_resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Error: {e}")

                with col_reject:
                    if st.button("❌ Reject", key=f"hitl_reject_{task_id}", use_container_width=True):
                        try:
                            review_resp = api_post(
                                api_url, f"/hitl/review/{task_id}",
                                json={"decision": "rejected", "comments": review_comments},
                                timeout=10,
                            )
                            if review_resp.status_code == 200:
                                st.success(f"❌ Task `{task_id}` rejected.")
                                st.rerun()
                            else:
                                st.error(f"Failed: {review_resp.status_code} - {review_resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Error: {e}")

                with col_msg:
                    # Placeholder for result messages
                    pass

                st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: Evaluation Dashboard
    # ═══════════════════════════════════════════════════════════════════
    with tab_eval:
        st.subheader("📊 Evaluation Dashboard")
        st.caption("Regression evaluation metrics and trends.")

        # Initialise session state
        if "eval_data" not in st.session_state:
            st.session_state.eval_data = None
        if "eval_fetch_error" not in st.session_state:
            st.session_state.eval_fetch_error = None

        col_fetch, col_status = st.columns([1, 4])
        with col_fetch:
            fetch_clicked = st.button("🔄 Fetch Evaluation", type="primary", key="eval_fetch")
        with col_status:
            status_placeholder = st.empty()

        if fetch_clicked:
            status_placeholder.info("Fetching regression evaluation...")
            st.session_state.eval_data = None
            st.session_state.eval_fetch_error = None
            try:
                resp = api_post(api_url, "/eval/regression", json={}, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        st.session_state.eval_data = data
                        status_placeholder.success("✅ Evaluation fetched successfully.")
                    else:
                        st.session_state.eval_fetch_error = data.get("error", "Unknown error")
                        status_placeholder.error(f"❌ {st.session_state.eval_fetch_error}")
                else:
                    st.session_state.eval_fetch_error = f"HTTP {resp.status_code}"
                    status_placeholder.error(f"❌ {st.session_state.eval_fetch_error}")
            except requests.RequestException as e:
                st.session_state.eval_fetch_error = str(e)
                status_placeholder.error(f"❌ Connection error: {e}")

        if st.session_state.eval_data:
            data = st.session_state.eval_data
            summary = data.get("summary", {})

            # Latest metric values
            st.markdown("### Latest Metric Values")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pass Rate", f"{summary.get('pass_rate', 0):.1%}" if summary.get("pass_rate") is not None else "N/A")
            with col2:
                st.metric("Total Cases", summary.get("total_cases", 0))
            with col3:
                st.metric("Passed", summary.get("passed_cases", 0))
            with col4:
                st.metric("Failed", summary.get("failed_cases", 0))

            # Trend charts via prepare_trend_data
            trend_data = prepare_trend_data([data])
            metrics = trend_data.get("metrics", [])
            available = trend_data.get("available_metrics", [])

            if metrics:
                df = pd.DataFrame(metrics)

                st.markdown("### Metric Trends")
                st.caption("Run evaluations over time to build trend history.")

                for metric_name in available:
                    if metric_name not in df.columns:
                        continue
                    col_chart, col_val = st.columns([3, 1])
                    with col_chart:
                        chart_df = df[[metric_name]].copy()
                        chart_df.index = df.get("timestamp", [""])
                        st.line_chart(chart_df, height=200)
                    with col_val:
                        latest_val = df[metric_name].iloc[-1]
                        if latest_val is not None:
                            st.metric(
                                metric_name.replace("_", " ").title(),
                                f"{latest_val:.2%}" if isinstance(latest_val, float) else str(latest_val),
                            )
                        else:
                            st.metric(metric_name.replace("_", " ").title(), "N/A")
                    st.divider()

                with st.expander("📋 View Raw Data", expanded=False):
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No evaluation history available")

        elif st.session_state.eval_fetch_error:
            st.error(f"Failed to fetch evaluation: {st.session_state.eval_fetch_error}")
            st.info("Ensure the backend is running and try again.")
        else:
            st.info("No evaluation history available")
            st.markdown("""
            Click **🔄 Fetch Evaluation** to run the regression suite and view:
            - **Pass rate** — overall pass/fail ratio
            - **Golden set pass rate** — pass rate on the golden dataset
            - **Answer stability** — consistency of responses across runs
            - **Regulatory compliance** — adherence to regulatory rules
            - **Role appropriateness** — correctness of role-based access
            - **HITL trigger precision** — accuracy of human-in-the-loop triggers
            """)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Prompt Versions
    # ═══════════════════════════════════════════════════════════════════
    with tab_prompts:
        st.subheader("📝 Prompt Version Manager")
        st.caption("Browse, compare, and activate prompt template versions.")

        # Fetch prompt list
        try:
            prompts_resp = api_get(api_url, "/prompts", timeout=10)
            if prompts_resp.status_code == 200:
                prompts_data = prompts_resp.json()
                prompt_list = prompts_data.get("prompts", {})
                prompt_names = sorted(prompt_list.keys())
            else:
                st.error(f"Failed to fetch prompts: {prompts_resp.status_code}")
                prompt_names = []
        except requests.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
            prompt_names = []

        if not prompt_names:
            st.info("No prompts found. Ensure the prompt manager is initialized on the backend.")
        else:
            # Prompt selector
            selected_prompt = st.selectbox(
                "Select Prompt",
                options=prompt_names,
                key="prompt_selector",
            )

            if selected_prompt:
                # Fetch version history for the selected prompt
                try:
                    history_resp = api_get(api_url, f"/prompts/{selected_prompt}/history", timeout=10)
                    if history_resp.status_code == 200:
                        history_data = history_resp.json()
                        versions = history_data.get("versions", [])
                        active_version = history_data.get("active_version", "")
                    else:
                        st.error(f"Failed to fetch history: {history_resp.status_code}")
                        versions = []
                        active_version = ""
                except requests.RequestException as e:
                    st.error(f"Error: {e}")
                    versions = []
                    active_version = ""

                if not versions:
                    st.info(f"No version history for '{selected_prompt}'.")
                else:
                    st.write(f"**Active Version:** `{active_version}`")
                    st.write(f"**Total Versions:** {len(versions)}")

                    # Display each version
                    for ver in versions:
                        ver_id = ver.get("version", "?")
                        author = ver.get("author", "unknown")
                        last_updated = ver.get("last_updated", "")
                        changelog = ver.get("changelog", {})
                        model_compat = ver.get("model_compatibility", [])
                        input_vars = ver.get("input_variables", [])
                        template = ver.get("template", "")
                        templates = ver.get("templates", {})
                        is_active = (ver_id == active_version)

                        with st.container(border=True):
                            # Version header
                            col_v, col_badge = st.columns([4, 1])
                            with col_v:
                                st.markdown(f"**Version:** `{ver_id}`")
                            with col_badge:
                                if is_active:
                                    st.success("✅ **ACTIVE**")
                                else:
                                    st.info("Inactive")

                            st.caption(f"Author: {author} | Last updated: {last_updated}")

                            # Changelog
                            if changelog:
                                with st.expander("📋 Changelog", expanded=False):
                                    if isinstance(changelog, dict):
                                        for key, val in changelog.items():
                                            st.write(f"**{key}:** {val}")
                                    else:
                                        st.write(str(changelog))

                            # Model compatibility
                            if model_compat:
                                st.write(f"**Models:** {', '.join(model_compat)}")

                            # Input variables
                            if input_vars:
                                st.write(f"**Input Variables:** {', '.join(input_vars)}")

                            # Template content
                            if template:
                                with st.expander("📄 Template Content", expanded=False):
                                    st.text_area(
                                        "Template",
                                        value=template,
                                        height=200,
                                        key=f"prompt_template_{selected_prompt}_{ver_id}",
                                        label_visibility="collapsed",
                                    )

                            # Multi-template (sub-templates)
                            if templates:
                                with st.expander("📄 Sub-Templates", expanded=False):
                                    for sub_key, sub_template in templates.items():
                                        st.markdown(f"**{sub_key}:**")
                                        st.text_area(
                                            f"Sub-template {sub_key}",
                                            value=sub_template,
                                            height=150,
                                            key=f"prompt_sub_{selected_prompt}_{ver_id}_{sub_key}",
                                            label_visibility="collapsed",
                                        )
                                        st.divider()

                            # Activate/Rollback button (only if not already active)
                            if not is_active:
                                if st.button(
                                    f"🔄 Activate Version {ver_id}",
                                    key=f"prompt_activate_{selected_prompt}_{ver_id}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    try:
                                        activate_resp = api_post(
                                            api_url, f"/prompts/{selected_prompt}/activate",
                                            json={"version": ver_id},
                                            timeout=10,
                                        )
                                        if activate_resp.status_code == 200:
                                            st.success(f"✅ Activated version `{ver_id}` for `{selected_prompt}`!")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {activate_resp.status_code} - {activate_resp.text}")
                                    except requests.RequestException as e:
                                        st.error(f"Error: {e}")

    # Document Management Panel
    st.sidebar.divider()
    st.sidebar.header("📄 Document Management")
    
    # Fetch and display indexed documents
    try:
        sources_resp = api_get(api_url, "/sources", timeout=10)
        if sources_resp.status_code == 200:
            sources_data = sources_resp.json()
            documents = sources_data.get("documents", [])
            doc_count = sources_data.get("count", 0)
            
            st.sidebar.write(f"**Indexed Documents:** {doc_count}")
            
            if documents:
                with st.sidebar.expander("View Documents", expanded=False):
                    for doc in documents:
                        doc_id = doc.get("doc_id", "unknown")
                        doc_type = doc.get("doc_type", "document")
                        source_path = doc.get("source_path", "")
                        
                        st.write(f"**{doc_id}**")
                        st.caption(f"Type: {doc_type}")
                        if source_path:
                            st.caption(f"Path: `{source_path}`")
                        
                        # Delete button for each document
                        if st.button(f"🗑️ Delete", key=f"del_{doc_id}"):
                            try:
                                del_resp = api_delete(api_url, f"/sources/{doc_id}", timeout=10)
                                if del_resp.status_code == 200:
                                    st.sidebar.success(f"Deleted {doc_id}")
                                    st.rerun()
                                else:
                                    st.sidebar.error(f"Failed to delete {doc_id}")
                            except requests.RequestException as e:
                                st.sidebar.error(f"Error: {e}")
                        
                        st.divider()
            else:
                st.sidebar.info("No documents indexed yet.")
    except requests.RequestException:
        st.sidebar.warning("Unable to fetch document list.")
    
    # Load document from file path
    with st.sidebar.expander("📂 Load Document from Path", expanded=False):
        st.write("Load a document from the server filesystem:")
        
        file_path = st.text_input(
            "File Path",
            placeholder="/path/to/document.pdf or data/knowledge_base/policies/...",
            help="Enter the absolute or relative path to the document file"
        )
        
        if st.button("Load Document", type="primary"):
            if file_path.strip():
                try:
                    # Read the file from the path
                    path = Path(file_path.strip())
                    if not path.exists():
                        st.sidebar.error(f"File not found: {file_path}")
                    else:
                        # Determine document type from extension
                        file_extension = path.suffix.lower()
                        if file_extension == ".pdf":
                            doc_type = "policy_wording"
                        elif file_extension in [".docx", ".doc"]:
                            doc_type = "policy_wording"
                        elif file_extension == ".txt":
                            doc_type = "memo"
                        elif file_extension == ".md":
                            doc_type = "memo"
                        elif file_extension == ".csv":
                            doc_type = "memo"
                        elif file_extension == ".json":
                            doc_type = "memo"
                        else:
                            doc_type = "document"
                        
                        # Read file content
                        try:
                            if file_extension == ".pdf":
                                # For PDF files, we'll need to use a PDF loader
                                # For now, show a message that PDF needs special handling
                                st.sidebar.warning("PDF files require the backend to handle loading. Use the ingest endpoint with extracted text.")
                            else:
                                # Read text-based files
                                content = path.read_text(encoding="utf-8")
                                doc_id = path.stem
                                
                                # Ingest the document
                                ingest_payload = {
                                    "documents": [
                                        {
                                            "id": doc_id,
                                            "content": content,
                                            "doc_type": doc_type,
                                            "path": str(path.absolute()),
                                        }
                                    ]
                                }
                                
                                ingest_resp = api_post(
                                    api_url, "/ingest",
                                    json=ingest_payload,
                                    timeout=30,
                                )
                                
                                if ingest_resp.status_code == 200:
                                    result = ingest_resp.json()
                                    job_id = result.get("job_id", "unknown")
                                    st.sidebar.success(f"Loaded {doc_id}! Job ID: {job_id}")
                                    st.rerun()
                                else:
                                    st.sidebar.error("Failed to load document")
                        except UnicodeDecodeError:
                            st.sidebar.error("Unable to read file. Please ensure it's a text-based file (PDF requires special handling).")
                        except Exception as e:
                            st.sidebar.error(f"Error reading file: {e}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
            else:
                st.sidebar.warning("Please enter a file path")
    
    # Upload new documents
    with st.sidebar.expander("📝 Upload Text Content", expanded=False):
        st.write("Upload text content to add to the knowledge base:")
        
        uploaded_content = st.text_area(
            "Document Content",
            height=200,
            placeholder="Paste your document content here..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            doc_id = st.text_input("Document ID", placeholder="my_document")
        with col2:
            doc_type = st.selectbox("Type", ["policy_wording", "memo", "document", "regulation"])
        
        if st.button("Ingest Document", type="primary"):
            if uploaded_content.strip():
                try:
                    ingest_payload = {
                        "documents": [
                            {
                                "id": doc_id if doc_id else f"doc_{uuid4().hex[:8]}",
                                "content": uploaded_content,
                                "doc_type": doc_type,
                                "path": f"uploaded_{doc_id if doc_id else 'document'}",
                            }
                        ]
                    }
                    
                    ingest_resp = api_post(
                        api_url, "/ingest",
                        json=ingest_payload,
                        timeout=30,
                    )
                    
                    if ingest_resp.status_code == 200:
                        result = ingest_resp.json()
                        job_id = result.get("job_id", "unknown")
                        st.sidebar.success(f"Document ingested! Job ID: {job_id}")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to ingest document")
                except requests.RequestException as e:
                    st.sidebar.error(f"Error: {e}")
            else:
                st.sidebar.warning("Please enter document content")


if __name__ == "__main__":
    main()