import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence

from app.chains.base_chain import (
    build_faq_prompt_with_history,
    format_examples_from_objects,
    get_chat_model,
)
from app.memory.sqlite_memory import append_message, get_history, get_message_count
from app.models.faq import FAQIntent, FAQResponse
from app.prompts.examples_store import Example, select_examples
from app.prompts.loader import get_json_format_instruction
from app.tools.guardrails import run_all_guardrails

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract the last JSON object from a text response."""
    try:
        idx = text.rfind("{")
        if idx == -1:
            return None
        payload = text[idx:]
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _format_history_for_prompt(history: List[BaseMessage]) -> str:
    """Format a list of BaseMessage history into a readable string for the prompt."""
    lines = []
    for msg in history:
        role = msg.type if hasattr(msg, "type") else type(msg).__name__.replace("Message", "").lower()
        content = msg.content if hasattr(msg, "content") else str(msg)
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


class FAQChain:
    """A LangChain-style chain for FAQ question answering with guardrails,
    semantic example selection, SQLite-backed memory, and structured JSON output.

    This chain preserves the behavior of Week 2 (guardrails, structured FAQResponse)
    while introducing LangChain patterns, semantic few-shot selection, and
    multi-turn conversation memory via SQLite.
    """

    def __init__(self, memory=None):
        self.model = get_chat_model()
        # memory is kept for backward compatibility but we use
        # the standalone module-level functions for simplicity

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    def _handle_simple_acknowledgment(self, user_message: str) -> Optional[FAQResponse]:
        """Handle simple acknowledgments and greetings without calling the LLM."""
        message_lower = user_message.lower().strip()
        simple_responses = {
            # Greetings
            "hi": "Hello! How can I assist you with your insurance claim today?",
            "hello": "Hello! How can I assist you with your insurance claim today?",
            "hey": "Hi there! How can I help you with your insurance needs?",
            "good morning": "Good morning! How can I assist you today?",
            "good afternoon": "Good afternoon! How can I help you?",
            "good evening": "Good evening! How can I assist you?",
            # Acknowledgments
            "ok": "You're welcome! Let me know if you need anything else.",
            "okay": "You're welcome! Let me know if you need anything else.",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thank": "You're welcome! Is there anything else I can help you with?",
            "great": "Glad to hear that! Let me know if you need further assistance.",
            "good": "Great! Feel free to ask if you have any other questions.",
            "yes": "Perfect! How can I assist you further?",
            "no": "No problem! Let me know if you need help with anything else.",
            "sure": "Great! What else can I help you with?",
            "alright": "Okay! Let me know if you need anything else.",
        }

        if message_lower in simple_responses:
            return FAQResponse(
                intent=FAQIntent.OTHER,
                category="greeting" if message_lower in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"} else "acknowledgment",
                confidence=1.0,
                answer_text=simple_responses[message_lower],
                reasoning="Simple greeting or acknowledgment detected",
                metadata={"simple_acknowledgment": True},
            )
        return None

    def _run_guardrails(self, user_message: str) -> Optional[FAQResponse]:
        """Run guardrails on the user message. Returns an FAQResponse if triggered."""
        guardrail_result = run_all_guardrails(user_message)
        if guardrail_result["triggered"]:
            failure = guardrail_result["failures"][0]
            return FAQResponse(
                intent=FAQIntent.OTHER,
                category="guardrail",
                confidence=1.0,
                answer_text=f"Guardrail engaged: {failure['details']}",
                reasoning=f"Rule: {failure['rule']}",
                metadata={"guardrail_triggered": True, "rule": failure["rule"]},
            )
        return None

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt_inputs(
        self,
        session_id: str,
        user_message: str,
        examples: List[Example],
    ) -> Dict[str, Any]:
        """Build the input dict for the ChatPromptTemplate.

        Retrieves conversation history from SQLite, formats it, and combines
        with the user message, examples, and JSON format instruction.
        """
        # Get conversation history
        history_messages = get_history(session_id)
        history_text = _format_history_for_prompt(history_messages)

        # Build example block
        example_dicts = format_examples_from_objects(examples)
        example_lines = []
        for ex in example_dicts:
            example_lines.append(f"User: {ex['user']}")
            example_lines.append(f"Assistant: {ex['assistant']}")
        example_block = "\n\n".join(example_lines)

        return {
            "json_instruction": get_json_format_instruction(),
            "example_block": example_block,
            "history": history_text,
            "user_message": user_message,
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> FAQResponse:
        """Parse the LLM response text into a structured FAQResponse."""
        json_obj = _extract_json_from_text(text)
        if not json_obj:
            raise ValueError("No JSON block found in model response.")

        intent_value = json_obj.get("intent", "OTHER")
        intent = FAQIntent.__members__.get(intent_value, FAQIntent.OTHER)

        return FAQResponse(
            intent=intent,
            category=json_obj.get("category", "general"),
            confidence=float(json_obj.get("confidence", 0.5)),
            answer_text=json_obj.get("answer_text", text.split("{")[0].strip()),
            reasoning=json_obj.get("reasoning"),
            metadata=json_obj.get("metadata", {}),
        )

    def _call_llm(self, messages: List[BaseMessage]) -> FAQResponse:
        """Call the LLM with a list of messages and parse the response."""
        result = None
        try:
            if hasattr(self.model, "generate"):
                result = self.model.generate(messages=[messages])
            elif hasattr(self.model, "predict_messages"):
                result = self.model.predict_messages(messages=[messages])
            else:
                result = self.model.invoke(messages)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.error("FAQChain model invocation failed: %s", exc, exc_info=True)
            return FAQResponse(
                intent=FAQIntent.OTHER,
                category="error",
                confidence=0.0,
                answer_text="I'm sorry, I was unable to generate a structured response. Please try again later.",
                reasoning=str(exc),
            )

        answer_text = None
        if hasattr(result, "generations") and result.generations:
            first = result.generations[0][0]
            msg = getattr(first, "message", None)
            if msg is not None:
                answer_text = msg.content
            else:
                answer_text = getattr(first, "text", None)

        if answer_text is None and hasattr(result, "content"):
            answer_text = result.content

        if answer_text is None:
            answer_text = str(result)

        answer_text = answer_text.strip()
        try:
            return self._parse_response(answer_text)
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("FAQChain response parsing failed: %s", exc, exc_info=True)
            return FAQResponse(
                intent=FAQIntent.OTHER,
                category="error",
                confidence=0.0,
                answer_text="I'm sorry, I was unable to generate a structured response. Please try again later.",
                reasoning=str(exc),
            )

    # ------------------------------------------------------------------
    # Chain execution
    # ------------------------------------------------------------------

    def _build_and_run_chain(
        self,
        session_id: str,
        user_message: str,
        examples: List[Example],
    ) -> FAQResponse:
        """Build the full message list and call the model."""
        prompt = build_faq_prompt_with_history()
        prompt_inputs = self._build_prompt_inputs(session_id, user_message, examples)

        # Format the prompt to get the message list
        prompt_value = prompt.format_prompt(**prompt_inputs)
        messages = prompt_value.to_messages()

        return self._call_llm(messages)

    def invoke(
        self,
        session_id: str,
        user_message: str,
        persist_history: bool = True,
    ) -> FAQResponse:
        """Invoke the FAQ chain for a given session and message.

        This is the main entry point. It:
        1. Checks for simple acknowledgments (no LLM call).
        2. Runs guardrails.
        3. Selects semantic few-shot examples.
        4. Builds the prompt with conversation history.
        5. Calls the LLM and parses the structured response.
        6. Persists the conversation to SQLite (if persist_history=True).

        Args:
            session_id: The unique session identifier.
            user_message: The user's current message.
            persist_history: Whether to save the conversation to SQLite.

        Returns:
            A structured FAQResponse.
        """
        # Step 1: Simple acknowledgments
        simple_response = self._handle_simple_acknowledgment(user_message)
        if simple_response:
            if persist_history:
                append_message(session_id, "user", user_message)
                append_message(session_id, "assistant", simple_response.answer_text)
            return simple_response

        # Step 2: Guardrails
        guardrail_response = self._run_guardrails(user_message)
        if guardrail_response:
            if persist_history:
                append_message(session_id, "user", user_message)
                append_message(session_id, "assistant", guardrail_response.answer_text)
            return guardrail_response

        # Step 3: Handle missing API key
        if self.model is None:
            placeholder_text = (
                "(No API key configured) The LangChain FAQ assistant is not available until OPENAI_API_KEY is set. "
                "Please configure the environment variable and try again."
            )
            faq_response = FAQResponse(
                intent=FAQIntent.OTHER,
                category="placeholder",
                confidence=0.0,
                answer_text=placeholder_text,
                reasoning="Development mode - no API key available",
            )
            if persist_history:
                append_message(session_id, "user", user_message)
                append_message(session_id, "assistant", faq_response.answer_text)
            return faq_response

        # Step 4: Select semantic few-shot examples
        examples = select_examples(user_message, k=3)

        # Step 5: Build prompt with history and call LLM
        faq_response = self._build_and_run_chain(session_id, user_message, examples)

        # Step 6: Persist conversation history
        if persist_history:
            append_message(session_id, "user", user_message)
            append_message(session_id, "assistant", faq_response.answer_text)

        return faq_response