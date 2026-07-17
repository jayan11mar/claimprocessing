from typing import Dict, Any, Optional
import os
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.prompts.faq_examples import get_faq_examples
from app.prompts.loader import (
    get_system_template,
    get_few_shot_examples,
    get_json_format_instruction,
)
from app.config import get_settings
from app.models.faq import FAQResponse, FAQIntent

logger = logging.getLogger(__name__)


def build_prompt(user_message: str) -> str:
    from app.prompt_manager.registry import get_registry
    examples = get_few_shot_examples()
    try:
        registry = get_registry()
        agent_template = registry.get_template("agent_system")
        examples_block = "\n".join(
            f"User: {ex['user']}\nAssistant: {ex['assistant']}\n"
            for ex in examples[:3]
        )
        return agent_template.format(
            json_instruction=get_json_format_instruction(),
            examples=examples_block,
            user_message=user_message,
        )
    except Exception:
        # Fallback to inline if registry not available
        parts = [
            "You are an insurance FAQ assistant. Provide a concise answer and then a JSON block.",
            "",
            get_json_format_instruction(),
            "",
            "--- Few-shot examples ---",
        ]
        
        for ex in examples[:3]:
            parts.append(f"User: {ex['user']}")
            parts.append(f"Assistant: {ex['assistant']}")
            parts.append("")
        
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")
        return "\n".join(parts)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        idx = text.rfind("{")
        if idx != -1:
            maybe = text[idx:]
            return json.loads(maybe)
    except json.JSONDecodeError:
        pass
    return None


def _parse_faq_response(text: str) -> Optional[FAQResponse]:
    json_obj = _extract_json_from_text(text)
    if not json_obj:
        return None
    
    try:
        intent_str = json_obj.get("intent", "OTHER")
        if isinstance(intent_str, str):
            try:
                intent = FAQIntent[intent_str]
            except KeyError:
                intent = FAQIntent.OTHER
        else:
            intent = intent_str
        
        response = FAQResponse(
            intent=intent,
            category=json_obj.get("category", "general"),
            confidence=float(json_obj.get("confidence", 0.5)),
            answer_text=json_obj.get("answer_text", text.split("{")[0].strip()),
            reasoning=json_obj.get("reasoning"),
            metadata=json_obj.get("metadata", {}),
        )
        return response
    except (TypeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse FAQResponse: {e}")
        return None


def call_faq_llm(user_message: str, retry: bool = True) -> Optional[FAQResponse]:
    load_dotenv()
    settings = get_settings()
    api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    
    if not api_key or api_key.startswith("your-"):
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.0,
            answer_text="(No API key configured) This is a placeholder answer about claims. Please configure OPENAI_API_KEY to get real responses.",
            reasoning="Development mode - no API key available",
        )

    openai_client = OpenAI(api_key=api_key)
    prompt = build_prompt(user_message)

    try:
        resp = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": get_system_template("main_faq_assistant")},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=1000,
        )
        text = resp.choices[0].message.content.strip()
        
        faq_response = _parse_faq_response(text)
        if faq_response:
            return faq_response
        
        if retry:
            logger.warning("First parse attempt failed, retrying with stricter instruction...")
            from app.prompt_manager.registry import get_registry
            try:
                registry = get_registry()
                retry_template = registry.get_template("faq_json_instruction")
                retry_prompt = (
                    f"Please respond ONLY with a valid JSON object matching this structure:\n"
                    f"{retry_template}\n\n"
                    f"User question: {user_message}"
                )
            except Exception:
                retry_prompt = (
                    f"Please respond ONLY with a valid JSON object matching this structure:\n"
                    f'{{"intent": "CLAIM_REGISTRATION", "category": "claims", "confidence": 0.8, '
                    f'"answer_text": "...", "reasoning": "..."}}\n\n'
                    f"User question: {user_message}"
                )
            retry_response = openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": get_system_template("main_faq_assistant")},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            text2 = retry_response.choices[0].message.content.strip()
            faq_response = _parse_faq_response(text2)
            if faq_response:
                return faq_response
        
        logger.error(f"Could not parse response into FAQResponse: {text}")
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.3,
            answer_text=text,
            reasoning="Could not parse structured response from LLM",
        )
    except OpenAIError as e:
        logger.error(f"LLM call failed: {e}")
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.0,
            answer_text=f"Error querying the FAQ system: {str(e)}",
            reasoning=f"OpenAIError: {type(e).__name__}",
        )
    except (ValueError, TypeError) as e:
        logger.error(f"LLM response parsing failed: {e}")
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.0,
            answer_text=f"Error processing the FAQ response: {str(e)}",
            reasoning=f"Exception: {type(e).__name__}",
        )
