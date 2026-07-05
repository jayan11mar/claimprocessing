from datetime import datetime
import asyncio
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.logging.json_logger import get_logger
from app.chains.agent_chain import AgentChain
from app.langsmith_integration import get_langsmith_trace_id
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.rag.chunkers import ChunkConfig, chunk_document
from app.rag.embeddings import get_embedding_fn
from app.rag.evaluation_harness import run_rag_evaluation
from app.rag.loaders import Document, load_documents_from_manifest
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.vectorstores import get_vector_store


logger = get_logger("app.api.server")

app = FastAPI(title="Claims Assistant API", version="0.1.0")
_START_TIME = time.time()

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
    store.add(all_chunks, embeddings)
    store.persist()
    _rag_vector_store = store


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

        retrieval_trace, citations = _extract_rag_metadata(faq_response.metadata)

        return ChatResponse(
            answer_text=faq_response.answer_text,
            structured=faq_response,
            chain_metadata=chain_metadata,
            retrieval_trace=retrieval_trace,
            citations=citations,
        )
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
def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    chunks = _get_retrieval_chunks()
    if not chunks:
        return {"query": req.query, "top_k": req.top_k, "results": [], "source_count": 0}

    results = hybrid_retrieve(chunks, req.query, k=req.top_k)
    serialized_results: List[Dict[str, Any]] = []
    for item in results:
        chunk = item.get("chunk")
        if chunk is None:
            continue
        serialized_results.append(
            {
                "chunk": chunk.to_dict(),
                "score": item.get("rerank_score", item.get("combined_score", 0.0)),
                "chunk_id": item.get("chunk_id"),
                "source_id": item.get("source_id"),
                "source_path": item.get("source_path"),
            }
        )

    return {
        "query": req.query,
        "top_k": req.top_k,
        "results": serialized_results,
        "source_count": len(_rag_documents) if _rag_documents else 0,
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
        _rebuild_rag_index()
        return {"status": "deleted", "doc_id": doc_id}
    return {"status": "not_found", "doc_id": doc_id}


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
    _ensure_components()
    _memory.clear_history(session_id)
    logger.info("reset_history", {"session_id": session_id})
    return {"status": "ok", "session_id": session_id}