from datetime import datetime
import asyncio
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.logging.json_logger import get_logger
from app.chains.agent_chain import AgentChain
from app.chains.router import lcel_router
from app.langsmith_integration import get_langsmith_trace_id
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.hitl.manager import get_hitl_manager
from app.hitl.models import HITLReviewRequest, HITLTask
from app.callbacks.logging_cb import LoggingCallbackHandler
from app.callbacks.tracing_cb import TracingCallbackHandler
from app.callbacks.metrics_cb import MetricsCallbackHandler
from langchain_core.runnables import RunnableConfig
from app.rag.chunkers import ChunkConfig, chunk_document
from app.rag.embeddings import get_embedding_fn
from app.rag.evaluation_harness import run_rag_evaluation
from app.rag.loaders import Document, load_documents_from_manifest
from app.rag.vectorstores import get_vector_store
from app.prompt_manager.registry import get_registry, initialize_prompts
from app.mcp.registry import get_registry as get_mcp_registry
from app.mcp.client import MCPClient, MCPClientPool, get_client_pool, reset_client_pool
from app.mcp.tool_adapter import discover_and_create_tools

# ── RBAC imports ────────────────────────────────────────────────────────
from app.rbac.auth import (
    extract_role_context_from_request,
    get_service_role_context,
)
from app.rbac.models import (
    PermissionMatrix,
    RoleContext,
    AnonymousContext,
    Role,
)
from app.rbac.filter import build_role_metadata_filter, clamp_top_k
from app.rbac.validator import validate_retrieval_results
from app.rbac.audit import audit_retrieval, audit_top_k_clamp

logger = get_logger("app.api.server")

app = FastAPI(title="Claims Processing & Settlement API", version="0.1.0")
_START_TIME = time.time()

# ── Initialize prompt registry on startup ─────────────────────────────
@app.on_event("startup")
async def startup_prompt_manager():
    registry = initialize_prompts()
    count = len(registry.list_prompts())
    logger.info("prompt_manager_initialized", {"prompt_count": count})

# Expose this module as `server` in builtins so tests can reference `server` directly
import sys as _sys, builtins as _builtins
_builtins.server = _sys.modules[__name__]


@app.middleware("http")
async def add_correlation_and_timing(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    request.state.correlation_id = correlation_id
    request.state.start_time = time.time()
    request.state.timings = {"llm_ms": 0, "tools": []}

    response = await call_next(request)
    latency_ms = int((time.time() - request.state.start_time) * 1000)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-Latency-Ms"] = str(latency_ms)
    return response


# ── RBAC Middleware ─────────────────────────────────────────────────────
# Attaches a RoleContext to every request based on the JWT in the
# Authorization header.  The context is stored on request.state.role_context
# and consumed by the pre-retrieval filter and post-retrieval validator.

@app.middleware("http")
async def rbac_middleware(request: Request, call_next):
    settings = get_settings()
    if settings.ENABLE_RBAC:
        role_context = extract_role_context_from_request(request)
        request.state.role_context = role_context
    else:
        # When RBAC is disabled, attach a service role context with
        # unrestricted access to all document types.  This ensures
        # existing tests and operations continue to work unchanged.
        request.state.role_context = get_service_role_context()
    response = await call_next(request)
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
    retrieval_trace: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []


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


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    include_metadata: bool = True


_memory = None
_agent_chain = None
_rag_documents: Dict[str, Document] = {}
_rag_vector_store: Any = None
_ingest_jobs: Dict[str, Dict[str, Any]] = {}
_rag_initialized = False

# API-level conversation caching to avoid redundant processing
# Cache structure: {session_id: {"response": ChatResponse, "timestamp": float, "message_hash": str}}
_conversation_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes cache TTL
_CACHE_MAX_SIZE = 1000  # Maximum number of cached sessions


def _ensure_rag_documents_loaded() -> None:
    """Ensure manifest documents are loaded into _rag_documents."""
    global _rag_documents, _rag_initialized
    if not _rag_initialized:
        _rag_initialized = True
        if not _rag_documents:
            try:
                manifest_docs = load_documents_from_manifest()
                for doc in manifest_docs:
                    if doc.source_id not in _rag_documents:
                        _rag_documents[doc.source_id] = doc
            except Exception:
                pass


def _ensure_components() -> None:
    global _memory, _agent_chain
    if _memory is None:
        _memory = SQLiteMemory()
    if _agent_chain is None:
        _agent_chain = AgentChain(memory=_memory)


def _fallback_response(user_message: str, error_info: Optional[str] = None) -> FAQResponse:
    return FAQResponse(
        intent=FAQIntent.OTHER,
        category="fallback",
        confidence=0.0,
        answer_text=(
            "Sorry, I couldn't process that request right now. "
            "Please try again with a simpler question or rephrase your request."
        ),
        reasoning="Fallback response after failed execution.",
        metadata={"fallback": True, "error_info": error_info, "original_input": user_message},
    )


def _invoke_with_retry(session_id: str, user_message: str, context: dict) -> FAQResponse:
    try:
        response = _agent_chain.invoke(session_id, user_message, context=context)
        if response.category == "error":
            logger.warning("chat_parse_error_retry", {"session_id": session_id, "category": response.category, "reasoning": response.reasoning})
            response = _agent_chain.invoke(session_id, user_message, context=context)
        if response.category == "error":
            return _fallback_response(user_message, error_info=response.reasoning or "parse_error")
        return response
    except Exception as exc:
        logger.exception("chat_invoke_exception", {"session_id": session_id, "error": str(exc)})
        try:
            response = _agent_chain.invoke(session_id, user_message, context=context)
            if response.category != "error":
                return response
        except Exception as exc2:
            logger.exception("chat_retry_exception", {"session_id": session_id, "error": str(exc2)})
        return _fallback_response(user_message, error_info=str(exc))


def _build_chunk_config() -> ChunkConfig:
    settings = get_settings()
    return ChunkConfig(
        chunk_size=int(getattr(settings, "CHUNK_SIZE", 800)),
        chunk_overlap=int(getattr(settings, "CHUNK_OVERLAP", 100)),
    )


def _infer_doc_type(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith((".md", ".txt", ".markdown")):
        return "policy_wording"
    if name.endswith(".json"):
        return "memo"
    if name.endswith(".csv"):
        return "memo"
    if name.endswith((".pdf", ".docx")):
        return "policy_wording"
    return "document"


def _serialize_document(document: Document) -> Dict[str, Any]:
    return {
        "doc_id": document.source_id,
        "source_path": document.source_path,
        "doc_type": document.doc_type,
        "insurance_type": document.insurance_type,
        "product_code": document.product_code,
        "product_name": document.product_name,
        "claim_type": document.claim_type,
        "raw_metadata": document.raw_metadata,
    }


def _ensure_rag_vector_store_loaded() -> None:
    """Load the persisted FAISS vector store from disk if not already loaded."""
    global _rag_vector_store
    if _rag_vector_store is not None:
        return
    store = get_vector_store(backend=get_settings().VECTOR_BACKEND)
    if store.index is not None and store.chunk_count > 0:
        _rag_vector_store = store
        logger.info("persistent_vector_store_loaded", {"chunk_count": store.chunk_count})
    else:
        logger.warning("persistent_vector_store_missing", {"chunk_count": store.chunk_count})


def _rebuild_rag_index() -> None:
    global _rag_vector_store
    if not _rag_documents:
        _rag_vector_store = None
        return

    config = _build_chunk_config()
    all_chunks: List[Any] = []
    for document in _rag_documents.values():
        chunks = chunk_document(document, config, use_semantic=True)
        all_chunks.extend(chunks)

    if not all_chunks:
        _rag_vector_store = None
        return

    embed_fn = get_embedding_fn()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_fn(texts)
    dimension = len(embeddings[0]) if embeddings else 1536
    store = get_vector_store(backend=get_settings().VECTOR_BACKEND, dimension=dimension)
    store.delete(ids=None)  # Clear existing FAISS contents before add to avoid duplicate vectors
    store.add(all_chunks, embeddings)
    store.persist()
    _rag_vector_store = store
    logger.info("index_rebuilt", {"chunk_count": len(all_chunks), "dimension": dimension})


def _get_retrieval_chunks() -> List[Any]:
    _ensure_rag_documents_loaded()
    if _rag_documents:
        config = _build_chunk_config()
        chunks: List[Any] = []
        for document in _rag_documents.values():
            chunks.extend(chunk_document(document, config, use_semantic=True))
        return chunks

    return []


def _collect_indexed_documents() -> List[Dict[str, Any]]:
    _ensure_rag_documents_loaded()
    docs: List[Dict[str, Any]] = []
    for document in _rag_documents.values():
        docs.append(_serialize_document(document))
    return docs


def _extract_rag_metadata(metadata: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(metadata, dict):
        return [], []
    retrieval_trace = metadata.get("retrieval_trace") or []
    citations = metadata.get("citations") or []
    if isinstance(retrieval_trace, dict):
        retrieval_trace = [retrieval_trace]
    if isinstance(citations, dict):
        citations = [citations]
    return list(retrieval_trace), list(citations)


@app.get("/health")
def health() -> Dict[str, Any]:
    uptime = time.time() - _START_TIME
    db_status = "unknown"
    try:
        _ensure_components()
        history_reader = getattr(_memory, "get_history", None)
        if callable(history_reader):
            history_reader("health_check_session_id")
        else:
            history_reader = getattr(_memory, "get_history_records", None)
            if callable(history_reader):
                history_reader("health_check_session_id")
        db_status = "ok"
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        db_status = f"error: {exc}"
    except AttributeError:
        db_status = "ok"

    settings = get_settings()
    vector_store_status = "empty"
    if _rag_vector_store is not None:
        vector_store_status = "ok"
    elif _rag_documents:
        vector_store_status = "ready"

    return {
        "status": "ok",
        "version": app.version,
        "uptime_seconds": int(uptime),
        "model": settings.OPENAI_MODEL_NAME,
        "temperature": settings.OPENAI_MODEL_TEMPERATURE,
        "db_status": db_status,
        "vector_store_status": vector_store_status,
        "document_count": len(_rag_documents) if _rag_documents else 0,
    }


def _get_cache_key(session_id: str, message: str) -> str:
    """Generate a cache key from session_id and message."""
    return hashlib.md5(f"{session_id}:{message}".encode()).hexdigest()


def _cleanup_expired_cache() -> None:
    """Remove expired entries from the conversation cache."""
    global _conversation_cache
    current_time = time.time()
    expired_keys = [
        key for key, value in _conversation_cache.items()
        if current_time - value.get("timestamp", 0) > _CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _conversation_cache[key]
    
    # Enforce max cache size by removing oldest entries
    # Use strict inequality to ensure we're at or below max size
    while len(_conversation_cache) > _CACHE_MAX_SIZE:
        sorted_items = sorted(_conversation_cache.items(), key=lambda x: x[1].get("timestamp", 0))
        if sorted_items:
            oldest_key, _ = sorted_items[0]
            del _conversation_cache[oldest_key]


def _invoke_lcel(request: Request, session_id: str, user_message: str, correlation_id: Optional[str] = None) -> ChatResponse:
    """Invoke the LCEL router and build a ChatResponse.

    Attaches logging, tracing, and metrics callbacks via ``RunnableConfig``.
    """
    from app.langsmith_integration import get_langsmith_trace_id
    trace_id = get_langsmith_trace_id()

    metrics_handler = MetricsCallbackHandler()
    callbacks = [
        LoggingCallbackHandler(session_id=session_id),
        metrics_handler,
    ]
    if trace_id:
        callbacks.append(TracingCallbackHandler(session_id=session_id, trace_id=trace_id))

    config = RunnableConfig(callbacks=callbacks)

    inputs = {
        "session_id": session_id,
        "user_message": user_message,
        "metadata": {
            "correlation_id": correlation_id,
            "timings": getattr(request.state, "timings", {"llm_ms": 0, "tools": []}),
        },
    }

    result = lcel_router.invoke(inputs, config=config)

    answer_text = result.get("answer_text", "")
    intent_str = result.get("intent", "OTHER")
    category = result.get("category", "general")
    confidence = result.get("confidence", 0.0)
    reasoning = result.get("reasoning")
    inner_metadata = result.get("metadata", {})
    citations = result.get("citations", [])
    retrieval_trace = result.get("retrieval_trace", [])
    lcel_chain_metadata = result.get("chain_metadata", {})
    metrics_report = metrics_handler.report()

    structured = FAQResponse(
        intent=FAQIntent(intent_str) if intent_str in FAQIntent.__members__ else FAQIntent.OTHER,
        category=category,
        confidence=confidence,
        answer_text=answer_text,
        reasoning=reasoning,
        metadata=inner_metadata,
    )

    latency_ms = int((time.time() - getattr(request.state, "start_time", time.time())) * 1000)
    settings = get_settings()

    chain_metadata = {
        "latency_ms": latency_ms,
        "lcel": True,
        "lcel_metrics": metrics_report,
        "llm_ms": lcel_chain_metadata.get("llm_ms", 0),
        "tool_timings": lcel_chain_metadata.get("tool_timings", []),
        "is_tool_augmented": lcel_chain_metadata.get("is_tool_augmented", False),
        "model": settings.OPENAI_MODEL_NAME,
        "temperature": settings.OPENAI_MODEL_TEMPERATURE,
    }
    if trace_id:
        chain_metadata["langsmith_trace_id"] = trace_id

    return ChatResponse(
        answer_text=answer_text,
        structured=structured,
        chain_metadata=chain_metadata,
        retrieval_trace=retrieval_trace,
        citations=citations,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    start = time.time()
    correlation_id = getattr(request.state, "correlation_id", None)
    log_meta: Dict[str, Any] = {
        "session_id": req.session_id,
        "user_message": req.message,
        "correlation_id": correlation_id,
    }

    settings = get_settings()

    # ── LCEL path (behind flag) ────────────────────────────────────────
    if settings.ENABLE_LCEL:
        try:
            response = _invoke_lcel(request, req.session_id, req.message, correlation_id)
            # Cache the response for future identical requests
            cache_key = _get_cache_key(req.session_id, req.message)
            _cleanup_expired_cache()
            _conversation_cache[cache_key] = {
                "response": response,
                "timestamp": time.time(),
                "message_hash": cache_key,
                "session_id": req.session_id,
            }
            return response
        except Exception as exc:
            logger.error("lcel_chat_error", {"error": str(exc), **log_meta})
            # Fall through to legacy path on LCEL failure

    # ── Legacy path (fallback) ─────────────────────────────────────────
    try:
        _ensure_components()
        
        # Check cache for identical requests (same session + same message)
        cache_key = _get_cache_key(req.session_id, req.message)
        _cleanup_expired_cache()
        
        cached_entry = _conversation_cache.get(cache_key)
        if cached_entry:
            cache_age = time.time() - cached_entry.get("timestamp", 0)
            if cache_age < _CACHE_TTL_SECONDS:
                logger.info("chat_cache_hit", {
                    "session_id": req.session_id,
                    "cache_key": cache_key,
                    "cache_age_ms": int(cache_age * 1000),
                })
                # Return cached response but update latency to reflect current request time
                cached_response = cached_entry["response"]
                cached_response.chain_metadata["latency_ms"] = int((time.time() - start) * 1000)
                cached_response.chain_metadata["cache_hit"] = True
                return cached_response
        
        context = {"correlation_id": correlation_id, "timings": request.state.timings}
        faq_response = _invoke_with_retry(req.session_id, req.message, context=context)

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

        retrieval_trace, citations = _extract_rag_metadata(faq_response.metadata)

        response = ChatResponse(
            answer_text=faq_response.answer_text,
            structured=faq_response,
            chain_metadata=chain_metadata,
            retrieval_trace=retrieval_trace,
            citations=citations,
        )
        
        # Cache the response for future identical requests
        _conversation_cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "message_hash": cache_key,
            "session_id": req.session_id,
        }
        
        return response
    except Exception as exc:
        logger.error("chat_error", {"error": str(exc), **log_meta})
        fallback = FAQResponse(
            intent=FAQIntent.OTHER,
            category="fallback",
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


class IngestRequest(BaseModel):
    documents: List[Dict[str, str]] = []


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    job_id = str(uuid4())

    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "message": "Preparing ingestion",
        "document_count": 0,
    }
    _ingest_jobs[job_id] = job

    try:
        if not req.documents:
            raise ValueError("No documents were provided for ingestion")

        job.update({"status": "running", "progress": 10, "message": "Processing documents"})
        for doc_dict in req.documents:
            raw_text = doc_dict.get("content", "").strip()
            if not raw_text:
                continue

            source_id = doc_dict.get("id") or os.path.splitext(doc_dict.get("path", "doc"))[0]
            if not source_id:
                source_id = f"upload_{uuid4().hex[:6]}"

            document = Document(
                text=raw_text,
                source_id=source_id,
                source_path=doc_dict.get("path", source_id),
                doc_type=doc_dict.get("doc_type", "document"),
                insurance_type=doc_dict.get("insurance_type", "unknown"),
                raw_metadata={"source": "api_upload"},
            )
            _rag_documents[document.source_id] = document

        job.update({"progress": 50, "message": "Building embeddings and index"})
        _rebuild_rag_index()

        job.update({
            "status": "completed",
            "progress": 100,
            "message": "Ingestion complete",
            "document_count": len(_rag_documents),
        })

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Ingestion accepted",
            "job": job,
        }
    except Exception as exc:
        job.update({"status": "failed", "progress": 100, "message": str(exc)})
        return {
            "status": "failed",
            "job_id": job_id,
            "message": str(exc),
            "job": job,
        }


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str) -> Dict[str, Any]:
    job = _ingest_jobs.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "unknown", "progress": 0, "message": "Job not found"}
    return job


@app.post("/retrieve")
def retrieve(request: Request, req: RetrieveRequest) -> Dict[str, Any]:
    _ensure_rag_vector_store_loaded()
    if _rag_vector_store is None:
        return {
            "query": req.query,
            "top_k": req.top_k,
            "results": [],
            "source_count": 0,
            "message": "Persistent vector index not found or empty. Run ingestion first.",
        }

    # ── RBAC: Extract role context ──────────────────────────────────────
    role_context = getattr(request.state, "role_context", AnonymousContext())
    settings = get_settings()
    rbac_start_ns = time.perf_counter_ns()

    # ── RBAC: Build pre-retrieval metadata filter ───────────────────────
    effective_k = req.top_k
    metadata_filter = None
    if settings.ENABLE_RBAC:
        effective_k = clamp_top_k(role_context, req.top_k)
        metadata_filter = build_role_metadata_filter(
            role_context, query=req.query, requested_k=req.top_k,
        )

        # Short-circuit: empty filter dict means no allowed doc types
        if metadata_filter == {}:
            elapsed_ms = round((time.perf_counter_ns() - rbac_start_ns) / 1_000_000, 3)
            audit_retrieval(
                role_context=role_context,
                query=req.query,
                requested_k=req.top_k,
                effective_k=0,
                pre_filter_count=0,
                post_validator_count=0,
                stripped_count=0,
                elapsed_ms=elapsed_ms,
                metadata_filter_used=None,
                fallback_triggered=False,
            )
            return {
                "query": req.query,
                "top_k": 0,
                "results": [],
                "source_count": 0,
                "message": "Access denied: no document types are permitted for your role.",
            }

    # ── Perform retrieval ───────────────────────────────────────────────
    retriever = _rag_vector_store.as_retriever(
        search_kwargs={"k": effective_k, "embedding_fn": get_embedding_fn()}
    )
    docs = retriever.invoke(req.query)

    # ── Serialize results as dicts for RBAC processing ──────────────────
    serialized_results: List[Dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        serialized_results.append(
            {
                "chunk": {
                    "text": doc.page_content if hasattr(doc, "page_content") else "",
                    "source_id": metadata.get("source_id", ""),
                    "source_path": metadata.get("source_path", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "insurance_type": metadata.get("insurance_type", ""),
                    "product_code": metadata.get("product_code"),
                    "product_name": metadata.get("product_name"),
                    "claim_type": metadata.get("claim_type"),
                    "section": metadata.get("section"),
                    "clause_id": metadata.get("clause_id"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "raw_metadata": metadata,
                },
                "score": 0.0,
                "chunk_id": metadata.get("chunk_id"),
                "source_id": metadata.get("source_id", ""),
                "source_path": metadata.get("source_path", ""),
                "doc_type": metadata.get("doc_type", ""),
            }
        )

    pre_validator_count = len(serialized_results)

    # ── RBAC: Post-retrieval validation ─────────────────────────────────
    stripped_count = 0
    if settings.ENABLE_RBAC:
        validated = validate_retrieval_results(
            serialized_results, role_context, query=req.query,
        )
        stripped_count = pre_validator_count - len(validated)
        serialized_results = validated

    elapsed_ms = round((time.perf_counter_ns() - rbac_start_ns) / 1_000_000, 3)

    # ── RBAC: Audit every retrieval ─────────────────────────────────────
    audit_retrieval(
        role_context=role_context,
        query=req.query,
        requested_k=req.top_k,
        effective_k=effective_k,
        pre_filter_count=pre_validator_count,
        post_validator_count=len(serialized_results),
        stripped_count=stripped_count,
        elapsed_ms=elapsed_ms,
        metadata_filter_used=metadata_filter,
        fallback_triggered=False,
    )

    logger.info("retrieve_completed", {
        "query": req.query,
        "top_k": req.top_k,
        "effective_k": effective_k,
        "result_count": len(serialized_results),
        "rbac_stripped": stripped_count,
    })
    return {
        "query": req.query,
        "top_k": effective_k,
        "results": serialized_results,
        "source_count": len(serialized_results),
    }


@app.post("/evaluate")
async def evaluate(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    golden_path = os.path.join(base_dir, "eval", "golden_set.json")
    cases = []
    if os.path.exists(golden_path):
        with open(golden_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        # golden_set.json has a nested structure: {"projects": [{"name": "...", "items": [...]}]}
        if isinstance(raw, dict):
            projects = raw.get("projects", [])
            for project in projects:
                items = project.get("items", [])
                for item in items:
                    cases.append({
                        "name": item.get("id", item.get("name", "")),
                        "query": item.get("query", ""),
                        "expected_keywords": item.get("expected_chunks", []),
                        "top_k": item.get("top_k", 3),
                        "claim_context": project.get("name"),
                    })
        elif isinstance(raw, list):
            cases = raw
    if not cases:
        cases = [
            {
                "name": "coverage lookup",
                "query": "What does the policy cover for hospital claims?",
                "expected_keywords": ["coverage", "hospital", "claim"],
                "top_k": 3,
            }
        ]

    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, _get_retrieval_chunks)
    report = await loop.run_in_executor(None, run_rag_evaluation, cases, chunks)
    return {
        "status": "ok",
        "report": report,
        "source_count": len(_rag_documents) if _rag_documents else 0,
    }


@app.get("/sources")
def list_sources() -> Dict[str, Any]:
    documents = _collect_indexed_documents()
    return {
        "documents": documents,
        "count": len(documents),
    }


@app.delete("/sources/{doc_id}")
def delete_source(doc_id: str) -> Dict[str, Any]:
    _ensure_rag_documents_loaded()
    if doc_id in _rag_documents:
        del _rag_documents[doc_id]
        # Also delete from vector store if it exists
        if _rag_vector_store is not None:
            try:
                _rag_vector_store.delete(ids=[doc_id])
            except Exception:
                pass
        _rebuild_rag_index()
        logger.info("source_deleted", {"doc_id": doc_id})
        return {"status": "deleted", "doc_id": doc_id}
    logger.warning("source_delete_not_found", {"doc_id": doc_id})
    return {"status": "not_found", "doc_id": doc_id}


@app.post("/sources/{doc_id}/reload")
def reload_source(doc_id: str) -> Dict[str, Any]:
    """Re-ingest a single source document by its doc_id.
    Deletes the old chunks from the vector store and re-ingests from the manifest.
    """
    _ensure_rag_documents_loaded()

    # Find the document from the manifest
    try:
        manifest_docs = load_documents_from_manifest()
    except Exception as exc:
        return {"status": "error", "doc_id": doc_id, "message": f"Failed to load manifest: {exc}"}

    target_doc = None
    for doc in manifest_docs:
        if doc.source_id == doc_id:
            target_doc = doc
            break

    if target_doc is None:
        return {"status": "error", "doc_id": doc_id, "message": f"Document '{doc_id}' not found in manifest"}

    # Delete old entry if it exists
    if doc_id in _rag_documents:
        del _rag_documents[doc_id]
        if _rag_vector_store is not None:
            try:
                _rag_vector_store.delete(ids=[doc_id])
            except Exception:
                pass

    # Re-load the document
    _rag_documents[doc_id] = target_doc

    # Rebuild the index
    _rebuild_rag_index()

    logger.info("source_reloaded", {"doc_id": doc_id})
    return {
        "status": "reloaded",
        "doc_id": doc_id,
        "message": f"Document '{doc_id}' has been re-ingested.",
        "document_count": len(_rag_documents),
    }


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


def _invalidate_session_cache(session_id: str) -> None:
    """Invalidate all cache entries for a given session_id."""
    global _conversation_cache
    # We need to check all entries since we can't reverse-engineer the session_id from the hash
    # In practice, this is called rarely (only on reset), so iterating is acceptable
    keys_to_remove = []
    for key, value in _conversation_cache.items():
        # The cache key is md5(session_id:message), so we can't directly extract session_id
        # We'll need to store session_id in the cache entry for proper invalidation
        if value.get("session_id") == session_id:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del _conversation_cache[key]


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = req.session_id
    _ensure_components()
    _memory.clear_history(session_id)
    _invalidate_session_cache(session_id)
    logger.info("reset_history", {"session_id": session_id})
    return {"status": "ok", "session_id": session_id}


# ── RBAC Endpoints ─────────────────────────────────────────────────────


@app.get("/roles")
def list_roles() -> Dict[str, Any]:
    """List all roles and their permissions from the RBAC permission matrix.

    Returns:
        Dict with role definitions that can be consumed by the UI to
        understand what each role can access.
    """
    matrix = PermissionMatrix.get_instance()
    roles_dict = {}
    for role_name, perms in matrix.roles.items():
        roles_dict[role_name] = {
            "display_name": perms.display_name,
            "description": perms.description,
            "allowed_doc_types": perms.allowed_doc_types,
            "allowed_insurance_types": perms.allowed_insurance_types,
            "max_retrieval_k": perms.max_retrieval_k,
            "requires_explicit_consent": perms.requires_explicit_consent,
        }
    return {
        "status": "ok",
        "enabled": get_settings().ENABLE_RBAC,
        "roles": roles_dict,
        "role_count": len(roles_dict),
    }


@app.get("/auth/context")
def auth_context(request: Request) -> Dict[str, Any]:
    """Return the current authentication context for the request.

    Useful for debugging and for frontend components to verify the
    active role and permissions.

    Returns:
        Dict with user_id, role, permissions, and JWT claims.
    """
    role_context = getattr(request.state, "role_context", AnonymousContext())
    return {
        "status": "ok",
        "user_id": role_context.user_id,
        "role": role_context.role,
        "is_authenticated": role_context.is_authenticated,
        "permissions": {
            "allowed_doc_types": role_context.allowed_doc_types,
            "max_retrieval_k": role_context.max_k,
            "requires_explicit_consent": role_context.requires_consent,
        },
        "rbac_enabled": get_settings().ENABLE_RBAC,
    }


# ── HITL Endpoints (gated behind ENABLE_HITL) ─────────────────────────


@app.get("/hitl/pending")
def hitl_pending() -> Dict[str, Any]:
    """List all pending HITL tasks.

    Returns an empty list when HITL is disabled.
    """
    settings = get_settings()
    if not settings.ENABLE_HITL:
        return {"status": "ok", "tasks": [], "count": 0, "enabled": False}

    manager = get_hitl_manager()
    tasks = manager.list_pending()
    return {
        "status": "ok",
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "count": len(tasks),
        "enabled": True,
    }


@app.post("/hitl/review/{task_id}")
def hitl_review(task_id: str, req: HITLReviewRequest) -> Dict[str, Any]:
    """Review (approve or reject) a pending HITL task.

    The task must be in ``pending`` status.  Once reviewed, the decision
    is persisted and the task is no longer returned by ``/hitl/pending``.
    """
    settings = get_settings()
    if not settings.ENABLE_HITL:
        raise HTTPException(
            status_code=503,
            detail="HITL is disabled. Set ENABLE_HITL=true to enable.",
        )

    manager = get_hitl_manager()
    task = manager.resume(task_id, req.decision, req.comments)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found or is not pending.",
        )

    logger.info(
        "hitl_review_completed",
        {
            "task_id": task_id,
            "decision": req.decision,
            "comments": req.comments,
        },
    )
    return {
        "status": "ok",
        "task": task.model_dump(mode="json"),
        "message": f"Task '{task_id}' has been {req.decision}.",
    }


@app.get("/hitl/task/{task_id}")
def hitl_get_task(task_id: str) -> Dict[str, Any]:
    """Get a single HITL task by ID (regardless of status)."""
    settings = get_settings()
    if not settings.ENABLE_HITL:
        raise HTTPException(
            status_code=503,
            detail="HITL is disabled. Set ENABLE_HITL=true to enable.",
        )

    manager = get_hitl_manager()
    task = manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )
    return {"status": "ok", "task": task.model_dump(mode="json")}


# ── MCP Integration Endpoints (gated behind ENABLE_MCP) ──────────────

_mcp_client_pool: Optional[MCPClientPool] = None
_mcp_langchain_tools: List[Any] = []


@app.on_event("startup")
async def startup_mcp():
    """Initialize MCP client pool and discover tools if ENABLE_MCP is True."""
    settings = get_settings()
    if not settings.ENABLE_MCP:
        logger.info("mcp_disabled", {"reason": "ENABLE_MCP is False"})
        return

    global _mcp_client_pool, _mcp_langchain_tools
    try:
        registry = get_mcp_registry()
        servers = registry.list_servers()
        logger.info("mcp_startup", {"server_count": len(servers), "servers": list(servers.keys())})

        pool = get_client_pool()
        _mcp_client_pool = pool

        # Create MCP clients for each server and do health checks
        for key, srv_def in servers.items():
            client = MCPClient(srv_def)
            pool.register(key, client)
            try:
                healthy = await client.health_check(force=True)
                logger.info("mcp_server_health",
                            {"server": key, "healthy": healthy})
            except Exception as exc:
                logger.warning("mcp_server_startup_health_failed",
                               {"server": key, "error": str(exc)})

        # Discover and create LangChain tools
        _mcp_langchain_tools = discover_and_create_tools(pool)
        logger.info("mcp_tools_discovered", {"tool_count": len(_mcp_langchain_tools)})

    except FileNotFoundError as exc:
        logger.warning("mcp_config_missing", {"error": str(exc)})
    except Exception as exc:
        logger.error("mcp_startup_error", {"error": str(exc)})


@app.on_event("shutdown")
async def shutdown_mcp():
    """Close all MCP clients on shutdown."""
    global _mcp_client_pool
    if _mcp_client_pool is not None:
        await _mcp_client_pool.close_all()
        logger.info("mcp_clients_closed")


class MCPToolInfo(BaseModel):
    server: str
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPToolsResponse(BaseModel):
    tools: List[MCPToolInfo]
    tool_count: int
    enabled: bool


@app.get("/mcp/tools", response_model=MCPToolsResponse)
def list_mcp_tools() -> MCPToolsResponse:
    """List all discovered MCP tools."""
    settings = get_settings()
    if not settings.ENABLE_MCP:
        return MCPToolsResponse(tools=[], tool_count=0, enabled=False)

    registry = get_mcp_registry()
    all_tools = registry.get_all_tools()
    tools_list = []
    for server_key, tool_schema in all_tools:
        tools_list.append(MCPToolInfo(
            server=server_key,
            name=tool_schema.name,
            description=tool_schema.description,
            input_schema=tool_schema.input_schema,
        ))
    return MCPToolsResponse(tools=tools_list, tool_count=len(tools_list), enabled=True)


class MCPInvokeRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}


class MCPInvokeResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool: str
    success: bool
    latency_ms: int


@app.post("/mcp/invoke", response_model=MCPInvokeResponse)
async def invoke_mcp_tool(req: MCPInvokeRequest) -> MCPInvokeResponse:
    """Invoke an MCP tool by name."""
    settings = get_settings()
    if not settings.ENABLE_MCP:
        return MCPInvokeResponse(
            tool=req.tool, success=False,
            error="MCP is disabled. Set ENABLE_MCP=true to enable.",
            latency_ms=0,
        )

    start = time.time()
    registry = get_mcp_registry()
    found = registry.find_tool(req.tool)
    if not found:
        return MCPInvokeResponse(
            tool=req.tool, success=False,
            error=f"Tool '{req.tool}' not found. Use /mcp/tools to list available tools.",
            latency_ms=int((time.time() - start) * 1000),
        )

    server_key, tool_schema, server_def = found
    pool = _mcp_client_pool or get_client_pool()
    client = pool.get(server_key)
    if client is None:
        client = MCPClient(server_def)
        pool.register(server_key, client)

    try:
        result = await client.invoke_tool(req.tool, req.arguments)
        latency_ms = int((time.time() - start) * 1000)
        logger.info("mcp_invoke_success", {
            "tool": req.tool, "server": server_key, "latency_ms": latency_ms,
        })
        return MCPInvokeResponse(
            tool=req.tool, result=result, success=True, latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        logger.error("mcp_invoke_error", {
            "tool": req.tool, "server": server_key, "error": str(exc), "latency_ms": latency_ms,
        })
        return MCPInvokeResponse(
            tool=req.tool, success=False,
            error=str(exc), latency_ms=latency_ms,
        )


# ── Evaluation Regression and Drift Endpoints (Week 8) ─────────────────


class RegressionRequest(BaseModel):
    golden_set_path: Optional[str] = None
    project_filter: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    baseline_path: Optional[str] = None


class RegressionResponse(BaseModel):
    status: str
    summary: Dict[str, Any] = {}
    comparison: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.post("/eval/regression", response_model=RegressionResponse)
def eval_regression(request: Request, req: RegressionRequest) -> RegressionResponse:
    """Run a full regression evaluation against the golden set.

    Results are written to ``reports/regression_report.json`` and returned
    inline.  When a *baseline_path* is provided, the response includes a
    before/after comparison with regression and improvement counts.

    This endpoint is designed for CI pipelines and can be called with
    minimal configuration — sensible defaults are used for all optional
    fields.
    """
    try:
        from eval.regression_suite import run_regression

        # Default output directory: reports/ in the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_dir = os.path.join(base_dir, "reports")

        report = run_regression(
            golden_set_path=req.golden_set_path,
            output_dir=output_dir,
            project_filter=req.project_filter,
            thresholds=req.thresholds,
            baseline_path=req.baseline_path,
        )

        logger.info("eval_regression_completed", {
            "total": report["summary"]["total_cases"],
            "passed": report["summary"]["passed_cases"],
            "failed": report["summary"]["failed_cases"],
            "pass_rate": report["summary"]["pass_rate"],
        })

        return RegressionResponse(
            status="ok",
            summary=report["summary"],
            comparison=report.get("comparison"),
        )
    except Exception as exc:
        logger.exception("eval_regression_error", {"error": str(exc)})
        return RegressionResponse(
            status="error",
            error=str(exc),
        )


class DriftRequest(BaseModel):
    baseline_path: Optional[str] = None
    current_path: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None


class DriftResponse(BaseModel):
    ok: bool = True
    summary: Dict[str, Any] = {}
    drift: Dict[str, Any] = {}
    error: Optional[str] = None


@app.post("/eval/drift", response_model=DriftResponse)
def eval_drift(request: Request, req: DriftRequest) -> DriftResponse:
    """Detect drift in evaluation metrics by comparing baseline vs. current.

    Calls :func:`eval.drift.load_and_compare` to load two regression report
    JSONs and produce a per-metric drift report.

    When a path is omitted sensible defaults are used:

    * **baseline** — ``reports/_baseline_summary.json`` (the Week 8 baseline)
    * **current**  — ``reports/regression_report.json`` (or a fresh
      ``run_regression()`` run if that file doesn't exist)

    Returns ``{"ok": true, "summary": {...}, "drift": {...}}`` on success.
    On error the response carries ``{"ok": false, "error": "...", "drift": {}}``
    with HTTP 200.
    """
    try:
        from eval.drift import load_and_compare

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # --- Resolve paths ---------------------------------------------------
        if req.baseline_path:
            baseline_path = req.baseline_path
        else:
            baseline_path = os.path.join(base_dir, "reports", "_baseline_summary.json")

        if req.current_path:
            current_path = req.current_path
        else:
            current_path = os.path.join(base_dir, "reports", "regression_report.json")
            if not os.path.isfile(current_path):
                # Fall back to a fresh regression run
                from eval.regression_suite import run_regression

                output_dir = os.path.join(base_dir, "reports")
                report = run_regression(output_dir=output_dir)
                current_path = os.path.join(output_dir, "regression_report.json")

        # --- Delegate to load_and_compare ------------------------------------
        result = load_and_compare(baseline_path, current_path, req.thresholds)

        # If load_and_compare returned an error dict, honour the ok=false
        # contract while still returning HTTP 200.
        if isinstance(result, dict) and "error" in result:
            logger.warning("eval_drift_failed", {"error": result["error"]})
            return DriftResponse(
                ok=False,
                error=result["error"],
            )

        drift = result  # plain drift report dict

        # Build a summary
        summary = {
            "baseline_source": baseline_path,
            "current_source": current_path,
            "metrics_compared": list(drift.keys()),
            "drifted_metrics": [m for m, v in drift.items() if v.get("drifted")],
        }

        logger.info("eval_drift_completed", {
            "drifted": summary["drifted_metrics"],
            "metrics": summary["metrics_compared"],
        })

        return DriftResponse(summary=summary, drift=drift)

    except Exception as exc:
        logger.exception("eval_drift_error", {"error": str(exc)})
        return DriftResponse(ok=False, error=str(exc))


# ── Prompt Management Endpoints ───────────────────────────────────────


class ActivatePromptRequest(BaseModel):
    version: str


@app.get("/prompts")
def list_prompts() -> Dict[str, Any]:
    """List all registered prompts with their active versions."""
    registry = get_registry()
    prompts = registry.list_prompts()
    return {
        "status": "ok",
        "prompts": prompts,
        "count": len(prompts),
    }


@app.get("/prompts/{name}/history")
def prompt_history(name: str) -> Dict[str, Any]:
    """Get version history for a specific prompt."""
    registry = get_registry()
    history = registry.get_version_history(name)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{name}' not found",
        )
    return {
        "status": "ok",
        "name": name,
        "active_version": registry.get_active_version(name),
        "versions": history,
        "count": len(history),
    }


@app.post("/prompts/{name}/activate")
def activate_prompt_version(name: str, req: ActivatePromptRequest) -> Dict[str, Any]:
    """Activate a specific version of a prompt (rollback).

    This completes in O(1) — just updating the active version pointer.
    """
    import time
    start = time.time()
    registry = get_registry()
    success = registry.activate_version(name, req.version)
    elapsed_ms = int((time.time() - start) * 1000)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{name}' with version '{req.version}' not found",
        )

    logger.info("prompt_activated", {
        "name": name,
        "version": req.version,
        "elapsed_ms": elapsed_ms,
    })

    return {
        "status": "ok",
        "name": name,
        "active_version": req.version,
        "elapsed_ms": elapsed_ms,
        "message": f"Activated version '{req.version}' for prompt '{name}'",
    }
