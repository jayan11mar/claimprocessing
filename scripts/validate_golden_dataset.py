#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.chains.simple_faq_llm import call_faq_llm
from app.memory.sqlite_memory import SQLiteMemory
from app.tools.claims_intake import register_and_validate_claim
from app.tools.fraud_detector import compute_fraud_score
from app.tools.guardrails import detect_pii, detect_prompt_injection, is_off_topic
from app.tools.settlement_calculator import calculate_settlement
from app.models.faq import FAQResponse


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_float(actual: float, expected: float, tolerance: float = 0.01) -> bool:
    return abs(actual - expected) <= tolerance


def validate_faq_item(item: Dict[str, Any]) -> bool:
    text = item["input"]["text"]
    expected = item["expected"]
    response = call_faq_llm(text)

    if response is None:
        print(f"FAIL [{item['id']}]: FAQ response was None")
        return False

    if not isinstance(response, FAQResponse):
        print(f"FAIL [{item['id']}]: FAQ response was not a FAQResponse instance")
        return False

    if expected.get("intent") and response.intent.value != expected["intent"]:
        print(
            f"FAIL [{item['id']}]: intent mismatch - expected {expected['intent']}, got {response.intent.value}"
        )
        return False

    if expected.get("category") and response.category != expected["category"]:
        print(
            f"FAIL [{item['id']}]: category mismatch - expected {expected['category']}, got {response.category}"
        )
        return False

    if expected.get("confidence_min") is not None and response.confidence < expected["confidence_min"]:
        print(
            f"FAIL [{item['id']}]: confidence below expected threshold - {response.confidence} < {expected['confidence_min']}"
        )
        return False

    if expected.get("answer_non_empty") and not response.answer_text.strip():
        print(f"FAIL [{item['id']}]: answer_text is empty")
        return False

    if expected.get("answer_contains"):
        if expected["answer_contains"].lower() not in response.answer_text.lower():
            print(
                f"FAIL [{item['id']}]: answer_text does not contain expected text '{expected['answer_contains']}'"
            )
            return False

    return True


def validate_claims_item(item: Dict[str, Any]) -> bool:
    payload = item["input"]
    expected = item["expected"]
    result = register_and_validate_claim(
        policy_number=payload["policy_number"],
        claim_amount=payload.get("claim_amount"),
        extra_info=payload.get("extra_info", {}),
    )

    if expected.get("is_eligible") is not None and result.is_eligible != expected["is_eligible"]:
        print(
            f"FAIL [{item['id']}]: eligibility mismatch - expected {expected['is_eligible']}, got {result.is_eligible}"
        )
        return False

    if expected.get("approved_amount") is not None:
        tolerance = expected.get("approved_amount_tolerance", 0.01)
        if not compare_float(result.approved_amount, expected["approved_amount"], tolerance):
            print(
                f"FAIL [{item['id']}]: approved_amount mismatch - expected {expected['approved_amount']} +/- {tolerance}, got {result.approved_amount}"
            )
            return False

    if expected.get("metadata_keys"):
        for key in expected["metadata_keys"]:
            if key not in result.metadata:
                print(f"FAIL [{item['id']}]: missing expected metadata key '{key}'")
                return False

    if expected.get("metadata_contains"):
        for key, value in expected["metadata_contains"].items():
            if result.metadata.get(key) != value:
                print(
                    f"FAIL [{item['id']}]: metadata['{key}'] mismatch - expected {value}, got {result.metadata.get(key)}"
                )
                return False

    return True


def validate_fraud_item(item: Dict[str, Any]) -> bool:
    claim_id = item["input"]["claim_id"]
    expected = item["expected"]
    result = compute_fraud_score(claim_id=claim_id)

    if expected.get("score") is not None:
        tolerance = expected.get("score_tolerance", 0.01)
        if not compare_float(result.score, expected["score"], tolerance):
            print(
                f"FAIL [{item['id']}]: score mismatch - expected {expected['score']} +/- {tolerance}, got {result.score}"
            )
            return False

    if expected.get("signals"):
        for signal in expected["signals"]:
            if signal not in result.signals:
                print(f"FAIL [{item['id']}]: missing expected fraud signal '{signal}'")
                print(f"  actual signals: {result.signals}")
                return False

    return True


def validate_settlement_item(item: Dict[str, Any]) -> bool:
    claim_id = item["input"]["claim_id"]
    expected = item["expected"]
    try:
        result = calculate_settlement(claim_id)
    except Exception as exc:
        print(f"FAIL [{item['id']}]: settlement calculation failed: {exc}")
        return False

    for field in ["gross_amount", "deductible", "depreciation_amount", "copay_amount", "approved_amount"]:
        if expected.get(field) is not None:
            tolerance = expected.get("score_tolerance", 0.01) if field == "approved_amount" else 0.01
            if not compare_float(getattr(result, field), expected[field], tolerance):
                print(
                    f"FAIL [{item['id']}]: {field} mismatch - expected {expected[field]} +/- {tolerance}, got {getattr(result, field)}"
                )
                return False

    if expected.get("notes_contains"):
        for phrase in expected["notes_contains"]:
            if not any(phrase.lower() in note.lower() for note in result.notes):
                print(f"FAIL [{item['id']}]: note phrase not found: '{phrase}'")
                print(f"  notes: {result.notes}")
                return False

    return True


def validate_memory_item(item: Dict[str, Any], memory: SQLiteMemory) -> bool:
    session_id = item["input"]["session_id"]
    expected = item["expected"]
    memory.clear_history(session_id)

    for entry in item["input"]["history"]:
        memory.append_message(session_id, entry["role"], entry["content"])

    records = memory.get_history_records(session_id)
    if expected.get("record_count") is not None and len(records) != expected["record_count"]:
        print(
            f"FAIL [{item['id']}]: record_count mismatch - expected {expected['record_count']}, got {len(records)}"
        )
        return False

    if expected.get("last_message") is not None:
        last_expected = expected["last_message"]
        if records[-1] != last_expected:
            print(
                f"FAIL [{item['id']}]: last message mismatch - expected {last_expected}, got {records[-1]}"
            )
            return False

    memory.clear_history(session_id)
    return True


def validate_guardrails_item(item: Dict[str, Any]) -> bool:
    text = item["input"]["text"]
    expected = item["expected"]
    results = [detect_pii(text), is_off_topic(text), detect_prompt_injection(text)]
    failures = [r for r in results if r["triggered"]]

    if expected.get("triggered") is False:
        if failures:
            print(f"FAIL [{item['id']}]: expected no guardrail trigger, but got {failures}")
            return False
        return True

    if expected.get("rule") is not None:
        if not any(r["rule"] == expected["rule"] and r["triggered"] for r in results):
            print(
                f"FAIL [{item['id']}]: expected guardrail rule {expected['rule']} to trigger, got {failures}")
            return False

    return True


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "data" / "golden_dataset"
    if not dataset_dir.exists():
        print(f"ERROR: golden dataset directory not found at {dataset_dir}")
        return 1

    memory = SQLiteMemory()
    failures = 0
    skipped = 0

    map_validators = {
        "faq": validate_faq_item,
        "claims": validate_claims_item,
        "fraud": validate_fraud_item,
        "settlement": validate_settlement_item,
        "memory": lambda item: validate_memory_item(item, memory),
        "guardrails": validate_guardrails_item,
    }

    for dataset_path in sorted(dataset_dir.glob("*.json")):
        category = dataset_path.stem
        validator = map_validators.get(category)
        if validator is None:
            print(f"SKIP: No validator configured for {dataset_path.name}")
            skipped += 1
            continue

        items = load_dataset(dataset_path)
        print(f"Running {len(items)} golden tests from {dataset_path.name}...")
        for item in items:
            if category == "faq" and not os.getenv("OPENAI_API_KEY"):
                print(f"SKIP [{item['id']}]: OPENAI_API_KEY not configured for FAQ validation")
                skipped += 1
                continue

            try:
                ok = validator(item)
            except Exception as exc:
                print(f"FAIL [{item['id']}]: exception during validation: {exc}")
                ok = False

            if not ok:
                failures += 1

    print("\nGolden dataset validation results")
    print("--------------------------------")
    print(f"Failures: {failures}")
    print(f"Skipped: {skipped}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
