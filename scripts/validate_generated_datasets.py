"""
Validate the generated eval and golden sets.
"""

import json
from pathlib import Path
from typing import Any, Dict


def validate_eval_set(eval_path: Path) -> Dict[str, Any]:
    """Validate the eval set structure and content."""
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    
    issues = []
    items = eval_set.get("items", [])
    
    # Check required fields
    required_fields = ["id", "query", "expected_keywords", "top_k", "difficulty"]
    for i, item in enumerate(items):
        for field in required_fields:
            if field not in item:
                issues.append(f"Item {i+1} missing required field: {field}")
        
        # Check query is not empty
        if not item.get("query", "").strip():
            issues.append(f"Item {i+1} has empty query")
        
        # Check expected_keywords is not empty
        if not item.get("expected_keywords"):
            issues.append(f"Item {i+1} has no expected_keywords")
        
        # Check difficulty is valid
        if item.get("difficulty") not in ["easy", "medium", "hard"]:
            issues.append(f"Item {i+1} has invalid difficulty: {item.get('difficulty')}")
    
    return {
        "valid": len(issues) == 0,
        "total_items": len(items),
        "issues": issues,
        "summary": {
            "easy": sum(1 for item in items if item.get("difficulty") == "easy"),
            "medium": sum(1 for item in items if item.get("difficulty") == "medium"),
            "hard": sum(1 for item in items if item.get("difficulty") == "hard"),
        }
    }


def validate_golden_set(golden_path: Path) -> Dict[str, Any]:
    """Validate the golden set structure and content."""
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    
    issues = []
    items = golden_set.get("items", [])
    
    # Check required fields
    required_fields = ["id", "query", "expected_answer", "expected_chunks", "difficulty"]
    for i, item in enumerate(items):
        for field in required_fields:
            if field not in item:
                issues.append(f"Item {i+1} missing required field: {field}")
        
        # Check query is not empty
        if not item.get("query", "").strip():
            issues.append(f"Item {i+1} has empty query")
        
        # Check expected_answer is not empty
        if not item.get("expected_answer", "").strip():
            issues.append(f"Item {i+1} has empty expected_answer")
        
        # Check expected_chunks is not empty
        if not item.get("expected_chunks"):
            issues.append(f"Item {i+1} has no expected_chunks")
        
        # Check difficulty is valid
        if item.get("difficulty") not in ["easy", "medium", "hard"]:
            issues.append(f"Item {i+1} has invalid difficulty: {item.get('difficulty')}")
    
    return {
        "valid": len(issues) == 0,
        "total_items": len(items),
        "issues": issues,
        "summary": {
            "easy": sum(1 for item in items if item.get("difficulty") == "easy"),
            "medium": sum(1 for item in items if item.get("difficulty") == "medium"),
            "hard": sum(1 for item in items if item.get("difficulty") == "hard"),
        }
    }


def main():
    """Main validation function."""
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print()
    
    # Validate eval set
    eval_path = Path(__file__).parent.parent / "eval" / "eval_set.json"
    print(f"Validating eval set: {eval_path}")
    eval_result = validate_eval_set(eval_path)
    print(f"  → Valid: {eval_result['valid']}")
    print(f"  → Total items: {eval_result['total_items']}")
    print(f"  → Difficulty distribution: {eval_result['summary']}")
    if eval_result['issues']:
        print(f"  → Issues: {len(eval_result['issues'])}")
        for issue in eval_result['issues'][:5]:
            print(f"    - {issue}")
    print()
    
    # Validate golden set
    golden_path = Path(__file__).parent.parent / "data" / "golden_dataset" / "rag_knowledge_base_golden.json"
    print(f"Validating golden set: {golden_path}")
    golden_result = validate_golden_set(golden_path)
    print(f"  → Valid: {golden_result['valid']}")
    print(f"  → Total items: {golden_result['total_items']}")
    print(f"  → Difficulty distribution: {golden_result['summary']}")
    if golden_result['issues']:
        print(f"  → Issues: {len(golden_result['issues'])}")
        for issue in golden_result['issues'][:5]:
            print(f"    - {issue}")
    print()
    
    # Overall result
    print("=" * 60)
    if eval_result['valid'] and golden_result['valid']:
        print("✓ Both datasets are valid!")
    else:
        print("✗ Validation failed - please review issues above")
    print("=" * 60)


if __name__ == "__main__":
    main()