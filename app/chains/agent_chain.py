import logging
import re
import time
from typing import Any, Dict, Optional

from app.chains.faq_chain import FAQChain
from app.langsmith_integration import record_span, start_trace
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.tools.claim_status_checker import check_claim_status
from app.tools.claims_intake import register_and_validate_claim
from app.tools.fraud_detector import compute_fraud_score
from app.tools.knowledge_retrieval import knowledge_retrieval
from app.tools.policy_checker import check_policy_status
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
                "name": "policy_checker",
                "func": check_policy_status,
                "description": "Check the status of a policy (active, lapsed, cancelled) and whether claims can be filed.",
            },
            {
                "name": "settlement_calculator",
                "func": calculate_settlement,
                "description": "Calculate a settlement breakdown for a claim considering deductible, copay, and sub-limits.",
            },
            {
                "name": "claim_status_checker",
                "func": check_claim_status,
                "description": "Look up the status and details of an existing claim from the claims database by claim ID.",
            },
            {
                "name": "knowledge_retrieval",
                "func": knowledge_retrieval,
                "description": "Search the knowledge base for policy terms, coverage details, exclusions, and regulatory information. Use this for questions about policy coverage, exclusions, waiting periods, and specific policy wordings.",
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

    def _extract_policy_number_from_history(self, session_id: str) -> str:
        """Extract the most recent policy number from conversation history."""
        if not self.memory:
            return ""
        
        history = self.memory.get_history(session_id)
        # Search from most recent to oldest
        for message in reversed(history):
            content = message.content if hasattr(message, "content") else str(message)
            policy_number = self._extract_policy_number(content)
            if policy_number:
                return policy_number
        return ""

    def _extract_claim_id_from_history(self, session_id: str) -> str:
        """Extract the most recent claim ID from conversation history."""
        if not self.memory:
            return ""
        
        history = self.memory.get_history(session_id)
        # Search from most recent to oldest
        for message in reversed(history):
            content = message.content if hasattr(message, "content") else str(message)
            claim_id = self._extract_claim_id(content)
            if claim_id:
                return claim_id
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

        # Fallback: match uppercase-letter-prefixed IDs but EXCLUDE policy number format (P + digits)
        match = re.search(r"\b(?!P\d{5,}\b)([A-Z]+\d{3,})\b", text, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def _extract_claim_amount(self, text: str) -> float:
        patterns = [
            r"claim amount(?: is| of)? \$?([0-9,]+(?:\.[0-9]{1,2})?)\$?",
            r"₹\s*([0-9,]+(?:\.[0-9]{1,2})?)",
            r"\$([0-9,]+(?:\.[0-9]{1,2})?)",
            r"([0-9,]+(?:\.[0-9]{1,2})?)\s*\$",
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

    def _load_session_context(self, session_id: str) -> Dict[str, Any]:
        """Read active session history and extract all known entities for reuse.

        This is the single, explicit path for loading multi-turn context before
        tool and follow-up decisions. It scans the entire conversation history
        for policy numbers, claim IDs, incident dates, and claim amounts so
        that follow-up turns in the same session can reuse previously supplied
        information without the user needing to re-enter it.

        Extraction works in two ways:
        1. Regex pattern matching on message text (policy numbers, claim IDs, etc.)
        2. Structured JSON metadata parsing from assistant messages (looking for
           tool_output fields that contain structured entities)
        """
        import json
        context: Dict[str, Any] = {}

        if not self.memory or not session_id:
            return context

        # Defensively handle memory objects that may not expose get_history
        # (e.g. test fakes). All production SQLiteMemory instances support it.
        try:
            history = self.memory.get_history(session_id)
        except (AttributeError, TypeError):
            return context

        # Extract entities from each message in chronological order so that
        # the *last* occurrence of each entity type ends up in the context
        # (most recently mentioned value wins).
        for message in history:
            content = message.content if hasattr(message, "content") else str(message)

            # --- Method 1: Regex pattern matching on raw text ---
            policy_number = self._extract_policy_number(content)
            if policy_number:
                context["policy_number"] = policy_number

            claim_id = self._extract_claim_id(content)
            if claim_id:
                context["claim_id"] = claim_id

            incident_date = self._extract_incident_date(content)
            if incident_date:
                context["incident_date"] = incident_date

            claim_amount = self._extract_claim_amount(content)
            if claim_amount > 0:
                context["claim_amount"] = claim_amount

            # --- Method 2: Structured JSON metadata parsing ---
            # Assistant messages may contain JSON (e.g., tool output) embedded
            # in the text, or may have metadata fields on the message object.
            # Try to extract structured entities from any JSON found in content.
            if hasattr(message, "type") and message.type == "assistant":
                try:
                    # Look for JSON blocks in the assistant response
                    json_objects = self._extract_json_objects(content)
                    for obj in json_objects:
                        if isinstance(obj, dict):
                            # Check for tool_output with claim_id or policy_number
                            tool_output = obj.get("tool_output") or obj.get("metadata", {}).get("tool_output", {})
                            if isinstance(tool_output, dict):
                                if tool_output.get("claim_id") and "claim_id" not in context:
                                    context["claim_id"] = tool_output["claim_id"]
                                if tool_output.get("policy_number") and "policy_number" not in context:
                                    context["policy_number"] = tool_output["policy_number"]

                            # Direct metadata fields
                            metadata = obj.get("metadata", {})
                            if isinstance(metadata, dict):
                                if metadata.get("claim_id") and "claim_id" not in context:
                                    context["claim_id"] = metadata["claim_id"]
                                if metadata.get("policy_number") and "policy_number" not in context:
                                    context["policy_number"] = metadata["policy_number"]

                            # Top-level fields in the JSON
                            if obj.get("claim_id") and "claim_id" not in context:
                                context["claim_id"] = obj["claim_id"]
                            if obj.get("policy_number") and "policy_number" not in context:
                                context["policy_number"] = obj["policy_number"]
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        if "incident_date" in context:
            context.setdefault("extra_info", {})
            context["extra_info"]["incident_date"] = context["incident_date"]

        return context

    def _extract_json_objects(self, text: str) -> list:
        """Extract all JSON objects from a text string."""
        import json
        objects = []
        stack = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                stack.append(index)
            elif char == "}":
                if not stack:
                    continue
                start = stack.pop()
                candidate = text[start:index + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    continue
        return objects

    def _record_tool_timing(self, tool_name: str, start: float, timings: Dict[str, Any], trace_id: Optional[str]) -> None:
        elapsed = int((time.time() - start) * 1000)
        timings["tools"].append({"tool": tool_name, "ms": elapsed})
        if trace_id:
            record_span(tool_name, {"ms": elapsed, "trace_id": trace_id})

    def _format_claim_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        if not result.is_eligible:
            answer_text = (
                f"Claim registration failed. Policy number: {result.policy_number}. "
                f"Eligible: {result.is_eligible}. "
                f"Estimated payable amount after deductible: ${result.approved_amount:.2f}."
            )
        else:
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

    def _format_policy_status_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        """Format the policy status response.
        
        Returns only the policy status message without additional details.
        Details are available in metadata for follow-up questions.
        """
        answer_text = result.message

        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=answer_text,
            reasoning=base.reasoning,
            metadata={
                **base.metadata,
                "tool": "policy_checker",
                "tool_output": result.to_dict(),
            },
        )

    def _format_claim_status_answer(self, base: FAQResponse, result: Any) -> FAQResponse:
        answer_text = result.message
        return FAQResponse(
            intent=base.intent,
            category=base.category,
            confidence=base.confidence,
            answer_text=answer_text,
            reasoning=base.reasoning,
            metadata={
                **base.metadata,
                "tool": "claim_status_checker",
                "tool_output": result.to_dict(),
            },
        )

    def _handle_claim_registration(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> FAQResponse:
        policy_number = (
            self._extract_policy_number(message)
            or intent.metadata.get("policy_number")
            or self._extract_policy_number_from_history(session_id or "")
        )
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

        if claim_amount <= 0:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="I need the claim amount to process your claim. Please provide the amount you're claiming (e.g., '$5000' or 'claim amount is 5000').",
                reasoning="Claim amount missing or invalid.",
                metadata={"tool": "claims_intake", "error": "claim_amount_missing"},
            )

        details = intent.metadata.get("extra_info", {}) or {}
        if "incident_date" not in details:
            incident_date = self._extract_incident_date(message)
            if incident_date:
                details["incident_date"] = incident_date
            elif intent.metadata.get("incident_date"):
                details["incident_date"] = intent.metadata["incident_date"]

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
        session_id: Optional[str] = None,
    ) -> FAQResponse:
        # Prefer regex extraction from the raw message text over LLM metadata,
        # because the LLM may hallucinate claim_id values.
        # Fall back to session context if not found in current message.
        claim_id = (
            self._extract_claim_id(message)
            or intent.metadata.get("claim_id")
            or self._extract_claim_id_from_history(session_id or "")
        )
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
        # Prefer regex extraction from the raw message text over LLM metadata,
        # because the LLM may hallucinate claim_id values.
        claim_id = self._extract_claim_id(message) or intent.metadata.get("claim_id")
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

    def _handle_policy_status(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> FAQResponse:
        policy_number = (
            self._extract_policy_number(message)
            or intent.metadata.get("policy_number")
            or self._extract_policy_number_from_history(session_id or "")
        )
        if not policy_number:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="I need a policy number to check its status. Please provide the policy ID or policy number.",
                reasoning="Policy number missing from status check request.",
                metadata={"tool": "policy_checker", "error": "policy_number_missing"},
            )

        start = time.time()
        result = check_policy_status(policy_number)
        self._record_tool_timing("policy_checker", start, timings, trace_id)
        return self._format_policy_status_answer(intent, result)

    def _handle_claim_status(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
    ) -> FAQResponse:
        # Prefer regex extraction from the raw message text over LLM metadata,
        # because the LLM may hallucinate claim_id values (e.g. "REGISTRATION")
        # that do not match the actual claim ID pattern.
        claim_id = self._extract_claim_id(message) or intent.metadata.get("claim_id")
        if not claim_id:
            return FAQResponse(
                intent=intent.intent,
                category=intent.category,
                confidence=intent.confidence,
                answer_text="I need a claim ID to check its status. Please provide the claim ID (e.g., C1001).",
                reasoning="Claim identifier missing from claim status request.",
                metadata={"tool": "claim_status_checker", "error": "claim_id_missing"},
            )

        start = time.time()
        result = check_claim_status(claim_id)
        self._record_tool_timing("claim_status_checker", start, timings, trace_id)
        return self._format_claim_status_answer(intent, result)

    def _infer_insurance_type(self, message: str) -> Optional[str]:
        """Infer insurance type from query using comprehensive keyword matching.
        
        Args:
            message: The user query string.
            
        Returns:
            "health", "motor", or None if ambiguous/unknown.
        """
        message_lower = message.lower()
        
        # Health-related keywords (ordered by specificity - multi-word first)
        health_keywords = [
            "day care procedure", "day care treatment", "pre-existing disease",
            "pre existing disease", "network hospital", "cashless claim",
            "room rent", "ambulance charges", "icu charges", "hospitalization",
            "hospitalisation", "ayush treatment", "new born", "organ donor",
            "health checkup", "medical expense", "medical bill", "inpatient",
            "outpatient", "domiciliary", "maternity", "wellness",
            "preventive", "consultation", "diagnosis", "prescription",
            "therapy", "rehabilitation", "day care", "daycare",
            "health", "medical", "hospital", "surgery", "treatment",
            "disease", "illness", "icu", "ambulance", "cashless",
            "doctor", "patient", "opd"
        ]
        
        # Motor-related keywords (ordered by specificity - multi-word first)
        # Note: Avoid overly generic terms like "car" that can match substrings in health terms
        motor_keywords = [
            "own damage", "third party", "third-party", "personal accident",
            "owner driver", "paid driver", "no claim bonus", "ncb",
            "accident damage", "total loss", "four wheeler", "two wheeler",
            "spare parts", "plate number",
            "motor", "vehicle", "bike", "idv",
            "garage", "engine", "tyre", "bumper", "windshield",
            "collision", "theft", "comprehensive", "fuel",
            "registration", "rc", "chassis", "traffic", "road",
            "driving", "repair", "consumables", "depreciation", "premium"
        ]
        
        # Count matches for each type (check multi-word keywords first)
        health_matches = []
        motor_matches = []
        
        # Check health keywords
        for kw in health_keywords:
            if kw in message_lower:
                health_matches.append(kw)
        
        # Check motor keywords
        for kw in motor_keywords:
            if kw in message_lower:
                motor_matches.append(kw)
        
        # Log detection results
        logger.info(
            "Metadata filter inference",
            extra={
                "query": message,
                "health_keywords_found": health_matches,
                "motor_keywords_found": motor_matches,
            }
        )
        
        # Conflict handling: if both detected, return None to retrieve across all policies
        if health_matches and motor_matches:
            logger.warning(
                "Conflicting insurance type signals detected in query: %s. "
                "Health keywords: %s, Motor keywords: %s. "
                "Setting insurance_type to None to retrieve across all policies.",
                message,
                health_matches,
                motor_matches,
            )
            return None
        
        # Return detected type
        if health_matches:
            logger.info(
                "Detected health insurance query. Trigger keyword: '%s'",
                health_matches[0],
            )
            return "health"
        
        if motor_matches:
            logger.info(
                "Detected motor insurance query. Trigger keyword: '%s'",
                motor_matches[0],
            )
            return "motor"
        
        # No clear signal
        logger.info("No insurance type confidently detected for query: %s", message)
        return None

    def _handle_knowledge_retrieval(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
    ) -> FAQResponse:
        """Handle knowledge base retrieval queries using RAG."""
        start = time.time()
        
        # Extract insurance type from message using improved inference
        insurance_type = self._infer_insurance_type(message)
        
        # Build metadata filter if insurance type detected
        metadata_filter = None
        if insurance_type:
            metadata_filter = {"insurance_type": insurance_type}
        
        # Defensive logging at metadata filter creation point
        logger.info(
            "Creating metadata filter",
            extra={
                "query": message,
                "detected_insurance_type": insurance_type,
                "metadata_filter": metadata_filter,
                "reason": f"Keyword-based detection: {insurance_type}" if insurance_type else "No clear insurance type detected",
            }
        )
        
        # Call knowledge retrieval tool
        result = knowledge_retrieval(
            query=message,
            top_k=3,
            claim_context=None,
            metadata_filter=metadata_filter,
        )
        
        self._record_tool_timing("knowledge_retrieval", start, timings, trace_id)
        
        # Extract answer and citations
        answer_text = result.get("answer_text", "No relevant information found in the knowledge base.")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0.5)
        
        # Build metadata with citations
        metadata = {
            **intent.metadata,
            "tool": "knowledge_retrieval",
            "tool_output": result,
            "citations": citations,
        }
        
        return FAQResponse(
            intent=intent.intent,
            category=intent.category,
            confidence=confidence,
            answer_text=answer_text,
            reasoning=intent.reasoning or "Retrieved from knowledge base using RAG",
            metadata=metadata,
        )

    def _handle_escalation(
        self,
        intent: FAQResponse,
        message: str,
        timings: Dict[str, Any],
        trace_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> FAQResponse:
        # Extract claim ID from message or session context
        claim_id = (
            self._extract_claim_id(message)
            or intent.metadata.get("claim_id")
            or self._extract_claim_id_from_history(session_id or "")
        )
        
        if claim_id:
            answer_text = (
                f"I've escalated claim {claim_id} for priority review. "
                f"The claim has been flagged as high priority. "
                f"A senior claims officer will review it within 24 hours."
            )
        else:
            answer_text = (
                "I've escalated your claim for priority review. "
                "A senior claims officer will review it within 24 hours."
            )

        return FAQResponse(
            intent=intent.intent,
            category=intent.category,
            confidence=intent.confidence,
            answer_text=answer_text,
            reasoning="Claim escalation request processed.",
            metadata={
                **intent.metadata,
                "tool": "escalation",
                "claim_id": claim_id,
            },
        )

    def invoke(self, session_id: str, user_message: str, context: dict = None) -> FAQResponse:
        context = context or {}
        timings = context.get("timings") if isinstance(context.get("timings"), dict) else {"llm_ms": 0, "tools": []}

        trace_name = f"agent_invoke:{session_id}"
        with start_trace(trace_name) as trace:
            trace_id = trace.get("trace_id") if isinstance(trace, dict) else None

            # ----- Step 1: Load session context from history before any tool decisions -----
            session_context = self._load_session_context(session_id)

            start = time.time()
            response = self.faq_chain.invoke(session_id, user_message, persist_history=False)
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

        # ----- Step 2: Merge session context into response metadata so tool handlers can reuse it -----
        # Only inject stable identifiers (policy_number, claim_id) from history.
        # Per-turn data (claim_amount, incident_date) may change every message and
        # should be extracted fresh from the current user message text. Incidentally
        # this also prevents cross-test session contamination when tests share an id.
        _persistent_keys = {"policy_number", "claim_id"}
        if isinstance(response.metadata, dict):
            for key in _persistent_keys:
                if key in session_context and (key not in response.metadata or not response.metadata.get(key)):
                    response.metadata[key] = session_context[key]
        else:
            response.metadata = {k: v for k, v in session_context.items() if k in _persistent_keys}

        # ----- Step 3: Route to the appropriate tool handler -----
        if response.intent == FAQIntent.CLAIM_REGISTRATION:
            response = self._handle_claim_registration(response, user_message, timings, trace_id, session_id)
        elif response.intent == FAQIntent.POLICY_STATUS:
            response = self._handle_policy_status(response, user_message, timings, trace_id, session_id)
        elif response.intent == FAQIntent.CLAIM_STATUS:
            response = self._handle_claim_status(response, user_message, timings, trace_id)
        elif response.intent == FAQIntent.FRAUD_CHECK:
            response = self._handle_fraud_check(response, user_message, timings, trace_id, session_id)
        elif response.intent == FAQIntent.SETTLEMENT_QUERY:
            response = self._handle_settlement_query(response, user_message, timings, trace_id)
        elif response.intent == FAQIntent.KNOWLEDGE_RETRIEVAL:
            response = self._handle_knowledge_retrieval(response, user_message, timings, trace_id)
        elif response.intent == FAQIntent.ESCALATION:
            response = self._handle_escalation(response, user_message, timings, trace_id, session_id)

        if isinstance(response.metadata, dict):
            response.metadata["timings"] = timings
        else:
            response.metadata = {"timings": timings}

        self.memory.append_message(session_id, "user", user_message)
        self.memory.append_message(session_id, "assistant", response.answer_text)
        if isinstance(context.get("timings"), dict):
            context["timings"].update(timings)

        return response