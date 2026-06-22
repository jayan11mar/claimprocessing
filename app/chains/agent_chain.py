import logging
import re
import time
from typing import Any, Dict, Optional

from app.chains.faq_chain import FAQChain
from app.langsmith_integration import record_span, start_trace
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.tools.claims_intake import register_and_validate_claim
from app.tools.fraud_detector import compute_fraud_score
from app.tools.settlement_calculator import calculate_settlement

logger = logging.getLogger(__name__)


class AgentChain:

    def __init__(self, memory: Optional[SQLiteMemory] = None):
        self.memory = memory or SQLiteMemory()
        self.faq_chain = FAQChain(memory=self.memory)
        self.tools = [
            {
                "name": "claims_intake",
                "func": register_and_validate_claim,
                "description": "Register a new claim and validate policy coverage details.",
            },
            {
                "name": "fraud_detector",
                "func": compute_fraud_score,
                "description": "Compute a fraud score for an existing claim based on policy and claim history.",
            },
            {
                "name": "settlement_calculator",
                "func": calculate_settlement,
                "description": "Calculate a settlement breakdown for a claim considering deductible, copay, and sub-limits.",
            },
        ]

    def _extract_policy_number(self, text: str) -> str:
        patterns = [
            r"\bP\d{5,}\b",
            r"policy(?: number| no|#)?\s*[:#]?\s*(P?\d{5,})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                policy_number = match.group(1).upper() if match.lastindex else match.group(0).upper()
                if not policy_number.startswith("P"):
                    policy_number = f"P{policy_number}"
                return policy_number
        return ""

    def _extract_incident_date(self, text: str) -> str:
        patterns = [
            r"(?:incident|loss|admission|treatment) date\s*(?:is|was|:)\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"(?:incident|loss|admission|treatment) date\s*(?:is|was|:)\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
            r"on\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
            r"on\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_supporting_documents(self, text: str) -> list[str]:
        documents = []
        patterns = [
            r"documents(?: required| needed| include)?[:]?\s*(.+)$",
            r"supporting documents(?: include)?[:]?\s*(.+)$",
            r"attached documents(?: are)?[:]?\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            parts = re.split(r",|;| and | & ", match.group(1))
            for part in parts:
                item = part.strip()
                if item:
                    documents.append(item)
        return documents

    def _extract_claim_id(self, text: str) -> str:
        match = re.search(r"\bC\d{3,}\b", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        match = re.search(r"claim\s+#?([A-Z0-9]{3,})", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r"\b([A-Z]+\d{3,})\b", text, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def _extract_claim_amount(self, text: str) -> float:
        patterns = [
            r"claim amount(?: is| of)? \$?([0-9,]+(?:\.[0-9]{1,2})?)",
            r"₹\s*([0-9,]+(?:\.[0-9]{1,2})?)",
            r"\$([0-9,]+(?:\.[0-9]{1,2})?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).replace(",", "")
            try:
                return float(value)
            except ValueError:
                continue
        return 0.0

    def _parse_claim_amount(self, value: Any, text: str) -> float:
        if value is None:
            return self._extract_claim_amount(text)

        if isinstance(value, str) and not value.strip():
            raise ValueError("Claim amount must be provided as a number.")

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Claim amount must be a numeric value.") from exc

    def _record_tool_timing(self, tool_name: str, start: float, timings: Dict[str, Any], trace_id: Optional[str]) -> None:
        elapsed = int((time.time() - start) * 1000)
        timings["tools"].append({"tool": tool_name, "ms": elapsed})
        if trace_id:
            record_span(tool_name, {"ms": elapsed, "trace_id": trace_id})

    def _format_claim_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        answer_text = (
            f"Claim registration completed. Claim ID: {result.claim_id}. "
            f"Eligible: {result.is_eligible}. "
            f"Estimated payable amount after deductible: ${result.approved_amount:.2f}."
        )
        if result.validation_messages:
            answer_text += " " + " ".join(result.validation_messages)

        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=answer_text,
            reasoning=base.reasoning,
            metadata={
                **base.metadata,
                "tool": "claims_intake",
                "tool_output": result.dict(),
            },
        )

    def _format_fraud_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        answer_text = (
            f"Fraud score for claim {result.claim_id}: {result.score:.2f}. "
            f"Signals: {', '.join(result.signals) if result.signals else 'None detected.'}"
        )
        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=answer_text,
            reasoning=base.reasoning,
            metadata={
                **base.metadata,
                "tool": "fraud_detector",
                "tool_output": result.dict(),
            },
        )

    def _format_settlement_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        answer_text = (
            f"Settlement for claim {result.claim_id}: gross amount ${result.gross_amount:.2f}, "
            f"deductible ${result.deductible:.2f}, copay ${result.copay_amount:.2f}, "
            f"approved amount ${result.approved_amount:.2f}."
        )
        if result.notes:
            answer_text += " " + " ".join(result.notes)

        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=answer_text,
            reasoning=base.reasoning,
            metadata={
                **base.metadata,
                "tool": "settlement_calculator",
                "tool_output": result.dict(),
            },
        )

    def _handle_claim_registration(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
    ) -> FAQResponse:
        policy_number = intent.metadata.get("policy_number") or self._extract_policy_number(message)
        if not policy_number:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="I need a valid policy number to register your claim. Please provide the policy ID or policy number.",
                reasoning="Policy number missing from claim registration request.",
                metadata={"tool": "claims_intake", "error": "policy_number_missing"},
            )

        try:
            claim_amount = self._parse_claim_amount(intent.metadata.get("claim_amount"), message)
        except ValueError as exc:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text=str(exc),
                reasoning="Invalid claim amount provided.",
                metadata={"tool": "claims_intake", "error": "invalid_claim_amount"},
            )

        details = intent.metadata.get("extra_info", {}) or {}
        if "incident_date" not in details:
            incident_date = self._extract_incident_date(message)
            if incident_date:
                details["incident_date"] = incident_date

        if "supporting_documents" not in details:
            documents = self._extract_supporting_documents(message)
            if documents:
                details["supporting_documents"] = documents

        start = time.time()
        claim = register_and_validate_claim(
            policy_number=policy_number,
            claim_amount=claim_amount,
            extra_info=details,
        )
        self._record_tool_timing("claims_intake", start, timings, trace_id)
        return self._format_claim_answer(intent, claim)

    def _handle_fraud_check(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
    ) -> FAQResponse:
        claim_id = intent.metadata.get("claim_id") or self._extract_claim_id(message)
        if not claim_id:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="Please provide a claim ID to review fraud indicators, for example C1001 or ABC123.",
                reasoning="Claim identifier missing from fraud check request.",
                metadata={"tool": "fraud_detector", "error": "claim_id_missing"},
            )

        start = time.time()
        fraud = compute_fraud_score(claim_id)
        self._record_tool_timing("fraud_detector", start, timings, trace_id)
        return self._format_fraud_answer(intent, fraud)

    def _handle_settlement_query(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
    ) -> FAQResponse:
        claim_id = intent.metadata.get("claim_id") or self._extract_claim_id(message)
        if not claim_id:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="I need a claim ID to calculate settlement. Please provide C1001 or ABC123.",
                reasoning="Claim identifier missing from settlement request.",
                metadata={"tool": "settlement_calculator", "error": "claim_id_missing"},
            )

        try:
            start = time.time()
            settlement = calculate_settlement(claim_id)
            self._record_tool_timing("settlement_calculator", start, timings, trace_id)
            return self._format_settlement_answer(intent, settlement)
        except ValueError as exc:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text=f"Unable to calculate settlement: {exc}",
                reasoning=str(exc),
                metadata={"tool": "settlement_calculator", "error": str(exc)},
            )

    def invoke(self, session_id: str, user_message: str, context: dict = None) -> FAQResponse:
        context = context or {}
        timings = context.get("timings") if isinstance(context.get("timings"), dict) else {"llm_ms": 0, "tools": []}

        trace_name = f"agent_invoke:{session_id}"
        with start_trace(trace_name) as trace:
            trace_id = trace.get("trace_id") if isinstance(trace, dict) else None

            start = time.time()
            response = self.faq_chain.invoke(session_id, user_message)
            timings["llm_ms"] = int((time.time() - start) * 1000)

            span_meta = {
                "component": "faq_chain",
                "session_id": session_id,
                "user_message_snippet": (user_message[:200] + "...") if len(user_message) > 200 else user_message,
                "llm_ms": timings["llm_ms"],
            }
            if trace_id:
                span_meta["trace_id"] = trace_id
            record_span("faq_chain", span_meta)

        if response.intent == FAQIntent.CLAIM_REGISTRATION:
            response = self._handle_claim_registration(response, user_message, timings, trace_id)
        elif response.intent == FAQIntent.FRAUD_CHECK:
            response = self._handle_fraud_check(response, user_message, timings, trace_id)
        elif response.intent == FAQIntent.SETTLEMENT_QUERY:
            response = self._handle_settlement_query(response, user_message, timings, trace_id)

        if isinstance(response.metadata, dict):
            response.metadata["timings"] = timings
        else:
            response.metadata = {"timings": timings}

        self.memory.append_message(session_id, "user", user_message)
        self.memory.append_message(session_id, "assistant", response.answer_text)
        if isinstance(context.get("timings"), dict):
            context["timings"].update(timings)

        return response
