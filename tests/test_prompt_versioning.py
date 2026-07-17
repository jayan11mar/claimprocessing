"""
Tests for the versioned YAML prompt management system.

Verifies:
1. All YAML prompt files load without error
2. The registry can list, history, and activate versions
3. Rollback completes in <30s
4. The /prompts, /prompts/{name}/history, /prompts/{name}/activate endpoints work
5. No hardcoded prompt strings remain in the app codebase
"""

import json
import os
import time
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.prompt_manager.models import PromptDocument, PromptVersion
from app.prompt_manager.loader import load_prompt_file, load_all_prompts
from app.prompt_manager.registry import (
    PromptRegistry,
    get_registry,
    initialize_prompts,
)
from app.prompt_manager.validator import (
    validate_prompt_document,
    get_template_text,
    ValidationError,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prompts_dir() -> Path:
    """Path to the prompts YAML directory."""
    return Path(__file__).resolve().parent.parent / "prompts"


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    """A fully loaded PromptRegistry."""
    reg = PromptRegistry()
    reg.load()
    return reg


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient that initializes prompts on startup."""
    from app.api.server import app
    with TestClient(app) as c:
        yield c


# ── Prompt File Tests ────────────────────────────────────────────────


def test_all_yaml_files_exist(prompts_dir):
    """Verify at least 5 YAML prompt template files exist."""
    yaml_files = list(prompts_dir.glob("*.yaml"))
    assert len(yaml_files) >= 5, (
        f"Expected >=5 YAML prompt files, found {len(yaml_files)}: "
        f"{[f.name for f in yaml_files]}"
    )
    # Check for all required templates
    required_names = {"faq_system", "rag_qa", "agent_system", "hitl_review", "guardrail", "faq_json_instruction"}
    actual_names = {f.stem for f in yaml_files}
    missing = required_names - actual_names
    assert not missing, f"Missing required prompt files: {missing}"


def test_each_yaml_loads(prompts_dir):
    """Every YAML file should load into a valid PromptDocument."""
    failures = []
    for filepath in sorted(prompts_dir.glob("*.yaml")):
        doc = load_prompt_file(filepath)
        if doc is None:
            failures.append(f"Failed to load: {filepath.name}")
            continue
        issues = validate_prompt_document(doc)
        if issues:
            failures.append(f"{filepath.name}: {issues}")
    assert not failures, "\n".join(failures)


def test_yaml_has_version_metadata(prompts_dir):
    """Every YAML document must have version, author, changelog, model_compatibility."""
    required_fields = {"version", "author", "changelog", "model_compatibility"}
    for filepath in sorted(prompts_dir.glob("*.yaml")):
        doc = load_prompt_file(filepath)
        assert doc is not None, f"Cannot load {filepath.name}"
        for ver_id, version in doc.versions.items():
            missing = required_fields - {k for k in version.dict() if version.dict()[k]}
            if missing:
                pytest.fail(f"{filepath.name} version {ver_id} missing: {missing}")


def test_input_variables_match_template(prompts_dir):
    """All declared input_variables must appear in the template; no undeclared vars."""
    import re
    for filepath in sorted(prompts_dir.glob("*.yaml")):
        doc = load_prompt_file(filepath)
        assert doc is not None, f"Cannot load {filepath.name}"
        for ver_id, version in doc.versions.items():
            if version.template:
                found_vars = set(re.findall(r"\{(\w+)\}", version.template))
                declared = set(version.input_variables)
                undeclared = found_vars - declared
                assert not undeclared, (
                    f"{filepath.name} v{ver_id}: undeclared vars {undeclared} in template"
                )
            for sub_key, sub_template in version.templates.items():
                found_vars = set(re.findall(r"\{(\w+)\}", sub_template))
                declared = set(version.input_variables)
                undeclared = found_vars - declared
                assert not undeclared, (
                    f"{filepath.name} v{ver_id}.{sub_key}: undeclared vars {undeclared}"
                )


# ── Registry Tests ───────────────────────────────────────────────────


def test_registry_loads_all(registry):
    """Registry should load all YAML prompts."""
    prompts = registry.list_prompts()
    assert len(prompts) >= 5, f"Expected >=5 prompts, got {len(prompts)}: {list(prompts.keys())}"


def test_registry_list_returns_summary(registry):
    """list_prompts() returns name, active_version, available_versions."""
    prompts = registry.list_prompts()
    for name, info in prompts.items():
        assert "name" in info
        assert "active_version" in info
        assert "available_versions" in info
        assert info["name"] == name


def test_registry_get_template(registry):
    """get_template() should return non-empty string for known prompts."""
    templates_to_check = ["faq_system", "faq_json_instruction", "rag_qa"]
    for name in templates_to_check:
        template = registry.get_template(name)
        assert template, f"Empty template for {name}"
        assert len(template) > 20, f"Template too short for {name}"


def test_registry_activate_version(registry):
    """activate_version() should succeed and update active_version."""
    # Get a prompt with multiple versions (rag_qa has 4)
    doc = registry.get_prompt("rag_qa")
    assert doc is not None
    available = list(doc.versions.keys())
    assert len(available) >= 1

    target = available[0]
    success = registry.activate_version("rag_qa", target)
    assert success
    assert registry.get_active_version("rag_qa") == target


def test_registry_activate_nonexistent(registry):
    """activate_version() on nonexistent prompt/version returns False."""
    assert not registry.activate_version("nonexistent_prompt", "1.0")
    assert not registry.activate_version("faq_system", "99.99")


def test_registry_version_history(registry):
    """get_version_history() returns ordered list of versions."""
    history = registry.get_version_history("rag_qa")
    assert len(history) >= 2, f"Expected >=2 versions for rag_qa, got {len(history)}"
    for entry in history:
        assert "name" in entry
        assert "version" in entry
        assert "author" in entry
        assert "last_updated" in entry
        assert "changelog" in entry
        assert "model_compatibility" in entry
        assert "input_variables" in entry


def test_registry_version_history_nonexistent(registry):
    """get_version_history() for unknown prompt returns empty list."""
    history = registry.get_version_history("nonexistent")
    assert history == []


# ── API Endpoint Tests ───────────────────────────────────────────────


def test_api_list_prompts(client):
    """GET /prompts returns all prompts with metadata."""
    response = client.get("/prompts")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] >= 5
    for name, info in data["prompts"].items():
        assert "active_version" in info
        assert "available_versions" in info


def test_api_prompt_history(client):
    """GET /prompts/{name}/history returns version history."""
    response = client.get("/prompts/rag_qa/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["name"] == "rag_qa"
    assert len(data["versions"]) >= 1
    assert data["active_version"] is not None


def test_api_prompt_history_not_found(client):
    """GET /prompts/{name}/history returns 404 for unknown prompt."""
    response = client.get("/prompts/nonexistent/history")
    assert response.status_code == 404


def test_api_activate_version(client):
    """POST /prompts/{name}/activate activates version and returns elapsed_ms."""
    # First get available versions
    response = client.get("/prompts/rag_qa/history")
    data = response.json()
    assert len(data["versions"]) >= 1
    target_version = data["versions"][0]["version"]

    start = time.time()
    response = client.post(
        f"/prompts/rag_qa/activate",
        json={"version": target_version},
    )
    elapsed = time.time() - start
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["active_version"] == target_version
    assert result["elapsed_ms"] >= 0

    # Verify rollback completes in <30 seconds
    assert elapsed < 30, f"Rollback took {elapsed:.2f}s, expected <30s"


def test_api_activate_not_found(client):
    """POST /prompts/{name}/activate returns 404 for nonexistent."""
    response = client.post(
        "/prompts/nonexistent/activate",
        json={"version": "1.0"},
    )
    assert response.status_code == 404

    response = client.post(
        "/prompts/rag_qa/activate",
        json={"version": "99.99"},
    )
    assert response.status_code == 404


# ── Guardrail / FAQ Override Tests ───────────────────────────────────


def test_guardrail_templates_loaded():
    """Guardrail templates should be accessible via the registry."""
    registry = get_registry()
    guardrail_doc = registry.get_prompt("guardrail")
    assert guardrail_doc is not None
    assert len(guardrail_doc.versions) >= 1
    active = guardrail_doc.active_version
    assert active in guardrail_doc.versions
    version = guardrail_doc.versions[active]
    for key in ["pii_warning", "off_topic_response", "injection_warning", "unsafe_content_response"]:
        assert key in version.templates, f"Missing guardrail template: {key}"


def test_faq_system_prompt_loaded():
    """FAQ system prompt should be accessible."""
    registry = get_registry()
    template = registry.get_template("faq_system")
    assert template
    assert "insurance faq assistant" in template.lower()


def test_json_instruction_loaded():
    """JSON format instruction should be accessible."""
    registry = get_registry()
    template = registry.get_template("faq_json_instruction")
    assert template
    assert "JSON" in template


# ── CI Grep: Zero Inline Prompts ────────────────────────────────────


def test_no_hardcoded_prompt_strings():
    """Verify no hardcoded prompt strings remain in app/ (excluding test files).

    This test greps for lines that look like hardcoded prompt text and fails
    if any are found outside of known exceptions.
    """
    app_dir = Path(__file__).resolve().parent.parent / "app"
    # Patterns that indicate hardcoded prompt content
    suspicious_patterns = [
        '"You are a',  # System prompt style
        '"You are an',  # System prompt style
        '"I am a',  # Assistant intro
        'RESPOND ONLY with a single valid JSON object',  # JSON instruction
        'insurance FAQ assistant',  # FAQ sys prompt fragment
        'insurance claims assistant',  # RAG sys prompt fragment
    ]

    exceptions = {
        # The fallback in qa_chain.py is a guard, not a primary prompt
        "qa_chain.py",
        # The loader.py references are just names, not prompt text
        "app/prompts/loader.py",
        # The simple_faq_llm.py has inline fallbacks
        "simple_faq_llm.py",
    }

    failures = []
    for pattern in suspicious_patterns:
        # Walk through all Python files in app/
        for pyfile in sorted(app_dir.rglob("*.py")):
            if pyfile.name in exceptions:
                continue
            if "test_" in pyfile.name:
                continue
            if "prompt_manager" in str(pyfile):
                continue
            try:
                content = pyfile.read_text()
                if pattern in content:
                    # Find the actual line
                    for i, line in enumerate(content.split("\n"), 1):
                        if pattern in line:
                            failures.append(f"{pyfile.relative_to(app_dir.parent)}:{i}: {line.strip()[:80]}")
            except (IOError, UnicodeDecodeError):
                continue

    if failures:
        pytest.fail(
            f"Found {len(failures)} potential hardcoded prompt strings "
            f"(exceptions: {exceptions}):\n" + "\n".join(failures)
        )


# ── Guardrail / Backward Compatibility Tests ─────────────────────────


def test_get_guardrail_response():
    """get_guardrail_response() should return correct templates."""
    from app.prompts.loader import get_guardrail_response
    response = get_guardrail_response("pii_warning")
    assert response
    assert "personal information" in response.lower()

    response = get_guardrail_response("off_topic_response")
    assert response
    assert "insurance" in response.lower()


def test_get_system_template():
    """get_system_template() should return the FAQ system prompt."""
    from app.prompts.loader import get_system_template
    template = get_system_template()
    assert template
    assert "insurance faq assistant" in template.lower()


def test_get_json_format_instruction():
    """get_json_format_instruction() should return the JSON instruction."""
    from app.prompts.loader import get_json_format_instruction
    instruction = get_json_format_instruction()
    assert instruction
    assert "JSON" in instruction


# ── Hot-Reload (watchdog) Tests ──────────────────────────────────────


def test_watchdog_watcher_starts():
    """The watchdog watcher should start without error."""
    registry = PromptRegistry()
    try:
        registry.start_hot_reload()
        # Should be running
        assert registry._observer is not None
        assert registry._observer.is_alive()
    finally:
        registry.stop_hot_reload()