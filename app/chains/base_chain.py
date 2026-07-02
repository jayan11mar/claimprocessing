from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from app.config import get_settings
from app.prompts.loader import get_system_template, get_json_format_instruction
from app.prompts.examples_store import Example


def get_chat_model() -> Optional[ChatOpenAI]:
    """Create and return an OpenAI chat model based on application settings."""
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


def format_examples_block(examples: List[Dict[str, str]]) -> str:
    """Format a list of example dicts into a string block for the prompt."""
    example_lines = []
    for example in examples:
        example_lines.append(f"User: {example['user']}")
        example_lines.append(f"Assistant: {example['assistant']}")
    return "\n\n".join(example_lines)


def format_examples_from_objects(examples: List[Example]) -> List[Dict[str, str]]:
    """Convert Example objects to dicts suitable for prompt formatting."""
    return [{"user": e.user, "assistant": e.assistant} for e in examples]


def build_faq_prompt(examples: List[Dict[str, str]]) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate for the FAQ chain with a system prompt and few-shot examples.

    Args:
        examples: List of dicts with 'user' and 'assistant' keys for few-shot examples.

    Returns:
        A ChatPromptTemplate ready for formatting.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(
        get_system_template("main_faq_assistant")
    )
    example_block = format_examples_block(examples)
    human_template = HumanMessagePromptTemplate.from_template(
        "{json_instruction}\n\n{example_block}\n\nUser: {user_message}\nAssistant:"
    )
    return ChatPromptTemplate.from_messages([system_prompt, human_template])


def build_faq_prompt_with_history() -> ChatPromptTemplate:
    """Build a ChatPromptTemplate that includes a placeholder for conversation history.

    The history is inserted as a formatted text block between the system message
    and the user message, allowing multi-turn context from SQLite memory to be
    injected seamlessly.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(
        get_system_template("main_faq_assistant")
    )
    human_template = HumanMessagePromptTemplate.from_template(
        "{json_instruction}\n\n{example_block}\n\n"
        "Conversation History (for context):\n{history}\n\n"
        "User: {user_message}\nAssistant:"
    )
    return ChatPromptTemplate.from_messages([system_prompt, human_template])