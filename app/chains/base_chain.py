from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from app.config import get_settings
from app.prompts.loader import get_system_template, get_json_format_instruction


def get_chat_model() -> Optional[ChatOpenAI]:
    settings = get_settings()
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        return None
    return ChatOpenAI(
        model=settings.OPENAI_MODEL_NAME,
        temperature=settings.OPENAI_MODEL_TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY,
        timeout=settings.OPENAI_REQUEST_TIMEOUT,
        max_retries=2,
    )


def build_faq_prompt(examples: List[Dict[str, str]]) -> ChatPromptTemplate:
    system_prompt = SystemMessagePromptTemplate.from_template(
        get_system_template("main_faq_assistant")
    )
    example_lines = []
    for example in examples:
        example_lines.append(f"User: {example['user']}")
        example_lines.append(f"Assistant: {example['assistant']}")
    example_block = "\n\n".join(example_lines)
    human_template = HumanMessagePromptTemplate.from_template(
        "{json_instruction}\n\n{example_block}\n\nUser: {user_message}\nAssistant:"
    )
    return ChatPromptTemplate.from_messages([system_prompt, human_template])
