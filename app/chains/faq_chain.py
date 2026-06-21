import json
import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, SystemMessage

from app.chains.base_chain import build_faq_prompt, get_chat_model
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.prompts.examples_store import select_examples
from app.prompts.loader import get_json_format_instruction
from app.tools.guardrails import run_all_guardrails

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[dict]:
    try:
        idx = text.rfind("{")
        if idx == -1:
            return None
        payload = text[idx:]
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


class FAQChain:

    def __init__(self, memory: Optional[SQLiteMemory] = None):
        self.model = get_chat_model()
        self.memory = memory or SQLiteMemory()

    def _build_messages(self, session_id: str, examples: List[dict], user_message: str) -> List[BaseMessage]:
        prompt = build_faq_prompt(examples)
        prompt_value = prompt.format_prompt(
            json_instruction=get_json_format_instruction(),
            example_block="\n\n".join(
                [f"User: {example['user']}\nAssistant: {example['assistant']}" for example in examples]
            ),
            user_message=user_message,
        )
        prompt_messages = prompt_value.to_messages()
        system_msgs = [m for m in prompt_messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in prompt_messages if not isinstance(m, SystemMessage)]
        history = self.memory.get_history(session_id)
        messages: List[BaseMessage] = []
        messages.extend(system_msgs)
        messages.extend(history)
        messages.extend(other_msgs)
        return messages

    def _parse_response(self, text: str) -> FAQResponse:
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

    def invoke(self, session_id: str, user_message: str, persist_history: bool = True) -> FAQResponse:
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
        else:
            examples = select_examples(user_message, k=3)
            messages = self._build_messages(session_id, [e.__dict__ for e in examples], user_message)

            try:
                result = None
                batch = [messages]
                if hasattr(self.model, "generate"):
                    result = self.model.generate(messages=batch)
                elif hasattr(self.model, "predict_messages"):
                    result = self.model.predict_messages(messages=batch)
                else:
                    result = self.model(messages=batch)

                answer_text = None
                if result is None:
                    raise RuntimeError("No response from model")

                gens = getattr(result, "generations", None)
                if gens:
                    first = gens[0][0]
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
                faq_response = self._parse_response(answer_text)
            except Exception as exc:
                logger.error("FAQChain failed: %s", exc, exc_info=True)
                faq_response = FAQResponse(
                    intent=FAQIntent.OTHER,
                    category="error",
                    confidence=0.0,
                    answer_text="I’m sorry, I was unable to generate a structured response. Please try again later.",
                    reasoning=str(exc),
                )

        if persist_history:
            self.memory.append_message(session_id, "user", user_message)
            self.memory.append_message(session_id, "assistant", faq_response.answer_text)
        return faq_response
