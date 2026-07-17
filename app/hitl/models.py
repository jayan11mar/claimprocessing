"""Pydantic models for HITL (Human-In-The-Loop) tasks."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class HITLTask(BaseModel):
    """A HITL task representing a paused action awaiting human review."""

    task_id: str = Field(default_factory=lambda: f"hitl_{uuid4().hex[:12]}")
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending | approved | rejected | expired

    # ── Trigger info ──────────────────────────────────────────────────
    rule_id: str
    rule_reason: str

    # ── Serialised recommendation with full context ───────────────────
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_trace: str = ""
    confidence: float = 0.0
    recommendation: Dict[str, Any] = Field(default_factory=dict)

    # ── Raw inputs for context ────────────────────────────────────────
    user_message: str = ""
    agent_response: str = ""

    # ── Review decision (populated on review) ─────────────────────────
    decision: Optional[str] = None  # approved | rejected
    reviewer_comments: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class HITLTriggerResult(BaseModel):
    """Result of evaluating HITL trigger rules against a request."""

    triggered: bool = False
    matched_rules: List[Dict[str, Any]] = Field(default_factory=list)
    task: Optional[HITLTask] = None


class HITLReviewRequest(BaseModel):
    """Request body for reviewing a HITL task."""

    decision: str = Field(..., pattern="^(approved|rejected)$")
    comments: Optional[str] = None