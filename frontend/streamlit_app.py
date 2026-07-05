import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import html
import streamlit as st
from typing import Any, Dict, Optional

import requests


def inject_chat_style() -> None:
    st.markdown(
        """
        <style>
        .chat-container { margin: 0; padding: 0; }
        .chat-message { margin: 8px 0; display: flex; width: 100%; }
        .chat-message--user { justify-content: flex-end; }
        .chat-message--assistant { justify-content: flex-start; }
        .chat-bubble {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 18px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .chat-message--user .chat-bubble {
            background: #0d6efd;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .chat-message--assistant .chat-bubble {
            background: #f2f3f5;
            color: #111;
            border-bottom-left-radius: 4px;
        }
        .chat-details {
            margin-top: 8px;
            font-size: 0.9rem;
            color: #555;
        }
        .chat-details summary {
            cursor: pointer;
            font-weight: 600;
        }
        .chat-details p { margin: 4px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_bubble(role: str, text: str, metadata: Optional[Dict[str, Any]] = None, index: int = 0) -> None:
    escaped_text = html.escape(text).replace("\n", "<br />")
    role_class = "user" if role == "user" else "assistant"
    title = "You" if role == "user" else "Assistant"
    st.markdown(
        f"""
        <div class='chat-container'>
            <div class='chat-message chat-message--{role_class}'>
                <div class='chat-bubble'><strong>{title}:</strong><br>{escaped_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # JSON context/details are now shown in the sidebar; do not display per-message metadata here.
    return


def get_backend_health(api_url: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None


def get_saved_history(api_url: str, session_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_url}/history/{session_id}", timeout=10)
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
    response = requests.post(
        f"{api_url}/chat",
        json=payload,
        timeout=30,
    )
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
        st.sidebar.warning("Backend health unavailable.")

    st.sidebar.divider()
    st.sidebar.header("LangSmith Tracing")
    if st.session_state.latest_langsmith_trace_id:
        st.sidebar.success(f"✓ Tracing enabled")
        st.sidebar.code(st.session_state.latest_langsmith_trace_id, language="text")
        st.sidebar.markdown(f"🔗 [View in LangSmith](https://smith.langchain.com)", unsafe_allow_html=True)
    else:
        st.sidebar.info("No trace ID yet. Send a message to generate a trace.")

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

    if hasattr(st, "chat_input"):
        query = st.chat_input("Ask a question about claims, coverage, or settlement")
    else:
        query = st.text_input("Ask a question about claims, coverage, or settlement")

    if query:
        try:
            resp = call_chat_api(st.session_state.session_id, query, api_url)
            structured = resp.get("structured", {})
            chain_metadata = resp.get("chain_metadata", {})
            st.session_state.latest_langsmith_trace_id = chain_metadata.get("langsmith_trace_id")
            st.session_state.history.append(
                {
                    "user": query,
                    "answer_text": resp.get("answer_text", ""),
                    "structured": structured,
                    "chain_metadata": chain_metadata,
                }
            )
        except requests.RequestException as exc:
            st.warning(f"Unable to reach backend: {exc}")
            st.session_state.history.append(
                {
                    "user": query,
                    "answer_text": "Sorry, the backend is not available. Please check the backend URL and try again.",
                    "structured": {},
                    "chain_metadata": {"error": str(exc)},
                }
            )

    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation History")
        st.write(f"**Local turns:** {len(st.session_state.history)}")

        for idx, item in enumerate(st.session_state.history):
            user_msg = item["user"]
            answer_text = item["answer_text"]
            structured = item.get("structured", {})
            chain_metadata = item.get("chain_metadata", {})

            render_chat_bubble("user", user_msg)
            render_chat_bubble("assistant", answer_text, None, idx)
            with st.expander("Response metadata", expanded=False):
                st.write("**Structured response**")
                st.json(structured)
                st.write("**Chain metadata**")
                st.json(chain_metadata)

    if st.button("Reset Conversation"):
        try:
            requests.post(
                f"{api_url}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=10,
            )
        except requests.RequestException:
            st.warning("Could not reset the session on the backend.")
        st.session_state.history = []
        st.rerun()


if __name__ == "__main__":
    main()