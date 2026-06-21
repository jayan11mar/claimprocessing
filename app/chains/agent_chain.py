import logging
import re
from typing import Any, Dict, Optional

from app.chains.faq_chain import FAQChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.tools.claims_intake import register_and_validate_claim
from app.tools.fraud_detector import compute_fraud_score
from app.tools.settlement_calculator import calculate_settlement
from app.langsmith_integration import start_trace, record_span
import time

logger = logging.getLogger(__name__)


class AgentChain:

    def __init__(self, memory: Optional[SQLiteMemory] = None):
        self.memory = memory or SQLiteMemory()
        self.faq_chain = FAQChain(memory=self.memory)
        self.tools = [
            {"name": "claims_intake", "func": register_and_validate_claim, "description": "Register a new claim and validate policy coverage details."},
            {"name": "fraud_detector", "func": compute_fraud_score, "description": "Compute a fraud score for an existing claim based on policy and claim history."},
            {"name": "settlement_calculator", "func": calculate_settlement, "description": "Calculate a settlement breakdown for a claim considering deductible, copay, and sub-limits."},
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
        docs = []
        patterns = [
            r"documents(?: required| needed| include)?[:]?\s*(.+)$",
            r"supporting documents(?: include)?[:]?\s*(.+)$",
            r"attached documents(?: are)?[:]?\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                parts = re.split(r",|;| and | & ", match.group(1))
                docs.extend([part.strip() for part in parts if part.strip()])
        return docs

    def _extract_claim_id(self, text: str) -> str:
        # Prefer explicit claim IDs like C1001, but also handle forms like ABC123
        match = re.search(r"\bC\d{3,}\b", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # common phrasing: "claim ABC123" or "claim #ABC123"
        match = re.search(r"claim\s+#?([A-Z0-9]{3,})", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # fallback: any token with letters then digits (e.g. ABC123)
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
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return 0.0

    def _attach_metadata(self, base: FAQResponse, tool_name: str, tool_result: Any) -> FAQResponse:
        metadata = {**base.metadata}
        metadata["tool"] = tool_name
        metadata["tool_output"] = tool_result.dict() if hasattr(tool_result, "dict") else tool_result
        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=base.answer_text,
            reasoning=base.reasoning,
            metadata=metadata,
        )

    def _format_claim_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        answer_text = (
            f"Claim registration completed. Claim ID: {result.claim_id}. "
            f"Eligible: {result.is_eligible}. "
            f"Estimated payable amount after deductible: ${result.approved_amount:.2f}."
        )
        if result.validation_messages:
            answer_text += " " + " ".join(result.validation_messages)
        response = FAQResponse(
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
        return response

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

    def invoke(self, session_id: str, user_message: str, context: dict = None) -> FAQResponse:
        context = context or {}
        timings = context.get("timings") if isinstance(context.get("timings"), dict) else {"llm_ms": 0, "tools": []}

        trace_name = f"agent_invoke:{session_id}"
        with start_trace(trace_name) as trace:
            trace_id = trace.get("trace_id") if isinstance(trace, dict) else None

            t0 = time.time()
            classification = self.faq_chain.invoke(session_id, user_message)
            llm_ms = int((time.time() - t0) * 1000)
            timings["llm_ms"] = llm_ms

            span_meta = {
                "component": "faq_chain",
                "session_id": session_id,
                "user_message_snippet": (user_message[:200] + "...") if len(user_message) > 200 else user_message,
                "llm_ms": llm_ms,
            }
            if trace_id:
                span_meta["trace_id"] = trace_id
            record_span("faq_chain", span_meta)

        if classification.intent == FAQIntent.CLAIM_REGISTRATION:
            try:
                policy_number = classification.metadata.get("policy_number") or self._extract_policy_number(user_message)
                if not policy_number:
                    final_response = FAQResponse(
                        intent=classification.intent,
                        category=classification.category,
                        confidence=classification.confidence,
                        answer_text="I need a valid policy number to register your claim. Please provide the policy ID or policy number.",
                        reasoning="Policy number missing from claim registration request.",
                        metadata={"tool": "claims_intake", "error": "policy_number_missing"},
                    )
                    return final_response

                claim_amount = float(classification.metadata.get("claim_amount") or self._extract_claim_amount(user_message))
                extra_info = classification.metadata.get("extra_info", {}) or {}
                if "incident_date" not in extra_info:
                    incident_date = self._extract_incident_date(user_message)
                    if incident_date:
                        extra_info["incident_date"] = incident_date
                if "supporting_documents" not in extra_info:
                    docs = self._extract_supporting_documents(user_message)
                    if docs:
                        extra_info["supporting_documents"] = docs

                t0 = time.time()
                claim_result = register_and_validate_claim(
                    policy_number=policy_number,
                    claim_amount=claim_amount,
                    extra_info=extra_info,
                )
                tool_ms = int((time.time() - t0) * 1000)
                timings["tools"].append({"tool": "claims_intake", "ms": tool_ms})
                if trace_id:
                    record_span("claims_intake", {"ms": tool_ms, "trace_id": trace_id})
                final_response = self._format_claim_answer(classification, claim_result)
            except Exception as exc:
                logger.error("claims_intake_error", exc_info=True)
                final_response = FAQResponse(
                    intent=classification.intent,
                    category=classification.category,
                    confidence=classification.confidence,
                    answer_text="I attempted to register a claim, but there was an error processing the intake.",
                    reasoning=str(exc),
                    metadata={"tool": "claims_intake", "error": str(exc)},
                )
        elif classification.intent == FAQIntent.FRAUD_CHECK:
            try:
                claim_id = classification.metadata.get("claim_id") or self._extract_claim_id(user_message)
                t0 = time.time()
                fraud_result = compute_fraud_score(claim_id)
                tool_ms = int((time.time() - t0) * 1000)
                timings["tools"].append({"tool": "fraud_detector", "ms": tool_ms})
                if trace_id:
                    record_span("fraud_detector", {"ms": tool_ms, "trace_id": trace_id})
                final_response = self._format_fraud_answer(classification, fraud_result)
            except Exception as exc:
                logger.error("fraud_detector_error", exc_info=True)
                final_response = FAQResponse(
                    intent=classification.intent,
                    category=classification.category,
                    confidence=classification.confidence,
                    answer_text="I attempted to compute fraud signals, but an error occurred.",
                    reasoning=str(exc),
                    metadata={"tool": "fraud_detector", "error": str(exc)},
                )
        elif classification.intent == FAQIntent.SETTLEMENT_QUERY:
            try:
                claim_id = classification.metadata.get("claim_id") or self._extract_claim_id(user_message)
                if not claim_id:
                    raise ValueError("No claim identifier found in the request. Please provide a claim id like C1001 or ABC123.")

                t0 = time.time()
                settlement_result = calculate_settlement(claim_id)
                tool_ms = int((time.time() - t0) * 1000)
                timings["tools"].append({"tool": "settlement_calculator", "ms": tool_ms})
                if trace_id:
                    record_span("settlement_calculator", {"ms": tool_ms, "trace_id": trace_id})
                final_response = self._format_settlement_answer(classification, settlement_result)
            except Exception as exc:
                logger.error("settlement_calculator_error", exc_info=True)
                err_msg = str(exc)
                user_msg = f"I attempted to calculate settlement for {claim_id}, but an error occurred: {err_msg}" if claim_id else f"I attempted to calculate settlement, but an error occurred: {err_msg}"
                final_response = FAQResponse(
                    intent=classification.intent,
                    category=classification.category,
                    confidence=classification.confidence,
                    answer_text=user_msg,
                    reasoning=err_msg,
                    metadata={"tool": "settlement_calculator", "error": err_msg},
                )
        else:
            final_response = classification

        try:
            if isinstance(final_response.metadata, dict):
                final_response.metadata["timings"] = timings
            else:
                final_response.metadata = {"timings": timings}
        except Exception:
            pass

        self.memory.append_message(session_id, "user", user_message)
        self.memory.append_message(session_id, "assistant", final_response.answer_text)
        if isinstance(context.get("timings"), dict):
            context["timings"].update(timings)

        return final_response
