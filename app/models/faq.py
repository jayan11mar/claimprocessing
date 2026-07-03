from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class FAQIntent(str, Enum):
    CLAIM_REGISTRATION = "CLAIM_REGISTRATION"
    POLICY_STATUS = "POLICY_STATUS"
    CLAIM_STATUS = "CLAIM_STATUS"
    FRAUD_CHECK = "FRAUD_CHECK"
    SETTLEMENT_QUERY = "SETTLEMENT_QUERY"
    DOCUMENTS_REQUIRED = "DOCUMENTS_REQUIRED"
    OTHER = "OTHER"


class FAQResponse(BaseModel):
    intent: FAQIntent
    category: str = Field(..., description="Category of the question (e.g., 'claims', 'policy', 'fraud')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    answer_text: str = Field(..., description="The actual answer text")
    reasoning: Optional[str] = Field(None, description="Optional reasoning for the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    class Config:
        use_enum_values = False
