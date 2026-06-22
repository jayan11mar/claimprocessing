from datetime import datetime
import sqlite3
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.logging.json_logger import get_logger
from app.chains.agent_chain import AgentChain
from app.langsmith_integration import get_langsmith_trace_id
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQResponse


logger = get_logger("app.api.server")

app = FastAPI(title="Claims Assistant API", version="0.1.0")
_START_TIME = time.time()

# Expose this module as `server` in builtins so tests can reference `server` directly
import sys as _sys, builtins as _builtins
_builtins.server = _sys.modules[__name__]


@app.middleware("http")
async def add_correlation_and_timing(request: Request, call_next):
    from uuid import uuid4

    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    request.state.correlation_id = correlation_id
    request.state.start_time = time.time()
    request.state.timings = {"llm_ms": 0, "tools": []}

    response = await call_next(request)
    latency_ms = int((time.time() - request.state.start_time) * 1000)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-Latency-Ms"] = str(latency_ms)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    corr = getattr(request.state, "correlation_id", None)
    log_meta = {"correlation_id": corr, "path": request.url.path}
    logger.exception("unhandled_exception", {**log_meta, "error": str(exc)})
    body = {"error": "Internal server error", "message": "An unexpected error occurred. Please try again later."}
    if corr:
        body["correlation_id"] = corr
    return JSONResponse(status_code=500, content=body)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer_text: str
    structured: FAQResponse
    chain_metadata: Dict[str, Any] = {}


class HistoryEntry(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    message_count: int
    turn_count: int
    history: List[HistoryEntry] = []


class ResetRequest(BaseModel):
    session_id: str


_memory = None
_agent_chain = None


def _ensure_components() -> None:
    global _memory, _agent_chain
    if _memory is None:
        _memory = SQLiteMemory()
    if _agent_chain is None:
        _agent_chain = AgentChain(memory=_memory)


@app.get("/health")
def health() -> Dict[str, Any]:
    uptime = time.time() - _START_TIME
    db_status = "unknown"
    try:
        _ensure_components()
        _memory.get_history("health_check_session_id")
        db_status = "ok"
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        db_status = f"error: {exc}"
    settings = get_settings()
    return {
        "status": "ok",
        "version": app.version,
        "uptime_seconds": int(uptime),
        "model": settings.OPENAI_MODEL_NAME,
        "temperature": settings.OPENAI_MODEL_TEMPERATURE,
        "db_status": db_status,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    start = time.time()
    correlation_id = getattr(request.state, "correlation_id", None)
    log_meta: Dict[str, Any] = {
        "session_id": req.session_id,
        "user_message": req.message,
        "correlation_id": correlation_id,
    }

    try:
        _ensure_components()
        context = {"correlation_id": correlation_id, "timings": request.state.timings}
        faq_response = _agent_chain.invoke(req.session_id, req.message, context=context)

        latency_ms = int((time.time() - start) * 1000)

        guardrail_flag = bool(faq_response.metadata.get("guardrail_triggered", False))
        error_info = faq_response.metadata.get("error_info") if isinstance(faq_response.metadata, dict) else None

        trace_id = get_langsmith_trace_id()

        tool_timings = request.state.timings.get("tools", [])
        tool_augmented = bool(tool_timings)
        expected_latency_ms = 8000 if tool_augmented else 3000

        log_meta.update({
            "intent": str(faq_response.intent),
            "confidence": float(faq_response.confidence),
            "latency_ms": latency_ms,
            "guardrail_triggered": guardrail_flag,
            "error_info": error_info,
            "llm_ms": request.state.timings.get("llm_ms", 0),
            "tool_timings": tool_timings,
            "is_tool_augmented": tool_augmented,
            "latency_target_ms": expected_latency_ms,
            "latency_within_target": latency_ms <= expected_latency_ms,
        })
        if trace_id:
            log_meta["langsmith_trace_id"] = trace_id

        if latency_ms > expected_latency_ms:
            logger.warning("latency_exceeded", {**log_meta, "latency_ms": latency_ms, "latency_target_ms": expected_latency_ms})

        logger.info("chat_handled", log_meta)

        settings = get_settings()
        chain_metadata = {
            "latency_ms": latency_ms,
            "llm_ms": request.state.timings.get("llm_ms", 0),
            "tool_timings": tool_timings,
            "is_tool_augmented": tool_augmented,
            "latency_target_ms": expected_latency_ms,
            "latency_within_target": latency_ms <= expected_latency_ms,
            "model": settings.OPENAI_MODEL_NAME,
            "temperature": settings.OPENAI_MODEL_TEMPERATURE,
        }
        if trace_id:
            chain_metadata["langsmith_trace_id"] = trace_id

        return ChatResponse(
            answer_text=faq_response.answer_text,
            structured=faq_response,
            chain_metadata=chain_metadata,
        )
    except Exception as exc:
        logger.error("chat_error", {"error": str(exc), **log_meta})
        fallback = FAQResponse(
            intent="OTHER",
            category="error",
            confidence=0.0,
            answer_text="Sorry, I encountered an error while processing your request. Please try again later.",
            reasoning=str(exc),
            metadata={"fallback": True, "error_info": str(exc)},
        )
        logger.info("chat_handled", {**log_meta, "intent": str(fallback.intent), "confidence": fallback.confidence, "latency_ms": 0, "guardrail_triggered": False, "error_info": str(exc)})
        resp = ChatResponse(
            answer_text=fallback.answer_text,
            structured=fallback,
            chain_metadata={
                "fallback": True,
                "latency_target_ms": 3000,
                "latency_within_target": True,
                "is_tool_augmented": False,
            },
        )
        return resp


@app.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str) -> HistoryResponse:
    _ensure_components()
    records = _memory.get_history_records(session_id)
    return HistoryResponse(
        session_id=session_id,
        message_count=len(records),
        turn_count=len(records) // 2,
        history=records,
    )


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = req.session_id
    _memory.clear_history(session_id)
    logger.info("reset_history", {"session_id": session_id})
    return {"status": "ok", "session_id": session_id}
