import re
from typing import Any, Dict


def detect_pii(text: str) -> Dict[str, Any]:
    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }
    
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, text):
            return {
                "triggered": True,
                "rule": "PII_DETECTED",
                "details": f"Found potential {pattern_name} in message.",
            }
    
    return {
        "triggered": False,
        "rule": "PII_DETECTED",
        "details": "",
    }


def is_off_topic(text: str) -> Dict[str, Any]:
    off_topic_keywords = [
        r"cook",
        r"recipe",
        r"sports",
        r"movie",
        r"music",
        r"politics",
        r"religion",
        r"weather",
        r"video game",
        r"technology repair",
        r"car maintenance",
        r"home repair",
    ]
    
    text_lower = text.lower()
    for keyword in off_topic_keywords:
        if keyword in text_lower:
            return {
                "triggered": True,
                "rule": "OFF_TOPIC",
                "details": "Question appears to be outside insurance/claims scope.",
            }
    
    return {
        "triggered": False,
        "rule": "OFF_TOPIC",
        "details": "",
    }


def detect_prompt_injection(text: str) -> Dict[str, Any]:
    injection_patterns = [
        r"ignore previous",
        r"system prompt",
        r"ignore instructions",
        r"act as",
        r"new instructions",
        r"you are now",
        r"disregard",
        r"override",
        r"pretend you are",
        r"from now on",
    ]
    
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return {
                "triggered": True,
                "rule": "PROMPT_INJECTION",
                "details": "Detected potential prompt injection attempt.",
            }
    
    return {
        "triggered": False,
        "rule": "PROMPT_INJECTION",
        "details": "",
    }


def run_all_guardrails(text: str) -> Dict[str, Any]:
    guardrails = [
        detect_pii(text),
        is_off_topic(text),
        detect_prompt_injection(text),
    ]
    
    failures = [g for g in guardrails if g["triggered"]]
    
    return {
        "triggered": len(failures) > 0,
        "failures": failures,
    }
