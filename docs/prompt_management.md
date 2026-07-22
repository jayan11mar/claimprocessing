# Prompt Management

## Purpose
This document describes the prompt management system, including the YAML prompt directory, version metadata, the prompt registry and loader architecture, available templates, and testing coverage.

## YAML Prompt Directory

All prompt templates are stored as versioned YAML files in **`config/prompts/`** — **6 files** containing **16+ distinct template versions**:

| File | Versions | Templates | Purpose |
|------|----------|-----------|---------|
| `rag_qa.yaml` | 4 versions (1.0, 0.3, 0.2, 0.1) | RAG QA with/without claim context | RAG question answering |
| `agent_system.yaml` | 1 version (1.0) | Agent system prompt | FAQ LLM agent |
| `faq_system.yaml` | 1 version (1.0) | FAQ assistant system | FAQ system instructions |
| `faq_json_instruction.yaml` | 1 version (1.0) | JSON format instruction | Structured JSON output |
| `guardrail.yaml` | 1 version (1.0) | 5 sub-templates | Guardrail responses (PII, off-topic, injection, unsafe, default refusal) |
| `hitl_review.yaml` | 1 version (1.0) | HITL review template | Human review notification |

### Version Metadata

Every YAML prompt file includes structured metadata:

```yaml
version: "1.0"
author: "system"
last_updated: "2026-07-17"
changelog:
  "1.0": "Initial prompt extracted from templates.json"
model_compatibility: ["gpt-4o-mini", "gpt-4", "gpt-4-turbo"]
input_variables:
  - context_str
  - query
  - claim_context
template: |
  ... template content ...
```

Metadata fields:
- `version` — Semantic version identifier
- `author` — Creator of the version
- `last_updated` — ISO date of last modification
- `changelog` — Per-version change descriptions
- `model_compatibility` — List of compatible models
- `input_variables` — Variables expected by the template
- `template` — The prompt template string (Jinja2-style `{variable}` placeholders)

Some files use multi-version format with `---` separators (e.g., `rag_qa.yaml` has 4 versions). Some use `templates` key for sub-templates (e.g., `guardrail.yaml` has 5 sub-templates).

## Architecture

### Prompt Registry (`app/prompt_manager/registry.py`)
- `PromptRegistry` class — versioned prompt storage and retrieval
- `get_prompt(name)` — Returns prompt document with all versions
- `get_template(name)` — Returns the active version's template string
- `initialize_prompts()` — Loads all YAML files into registry on startup
- Singleton via `get_registry()`

### YAML Loader (`app/prompt_manager/loader.py`)
- Reads YAML files from `config/prompts/` directory
- Parses multi-version documents (separated by `---`)
- Extracts metadata and templates

### Backward-Compatible Loader (`app/prompts/loader.py`)
- `load_templates()` — Loads templates via registry, maintaining old dict structure
- `get_system_template()` — Returns FAQ system prompt
- `get_guardrail_response()` — Returns guardrail response by rule name
- `get_few_shot_examples()` — Returns examples from legacy `templates.json`
- `get_json_format_instruction()` — Returns JSON format instruction

### Prompt API Endpoints (`app/api/server.py`)
- `GET /prompts` — List all registered prompts
- `GET /prompts/{name}/history` — Get version history for a prompt
- `POST /prompts/{name}/activate` — Activate a specific version

### Prompt Versions UI (`app/frontend/streamlit_app.py`, lines 838-975)
- Tab "📝 Prompt Versions" in Streamlit app
- Select prompt from dropdown
- View version history with expandable template content
- Activate/Rollback to any version

## Remaining Hardcoded Prompts

There is **1** hardcoded string in the codebase (all others resolve from YAML):

**File:** `app/prompts/loader.py`, line 28:
```python
_LAST_RESORT_REFUSAL = "I cannot assist with that request."
```

This is only reached if the YAML registry itself raises an exception (YAML parse error, missing file, or malformed document). Under normal operation, all prompts load from YAML via the registry.

## Test Evidence

- **Test file:** `tests/test_prompts_loader.py` — 12 test functions
- **Test file:** `tests/test_prompt_versioning.py` — 24 test functions
- Coverage includes: YAML loading, version metadata parsing, template retrieval, fallback behavior, multi-version documents, sub-template extraction

## Reviewer Demo

```bash
# List all prompt templates with their versions
python -c "
from pathlib import Path
import yaml
prompts_dir = Path('config/prompts')
for f in sorted(prompts_dir.glob('*.yaml')):
    with open(f) as fh:
        content = fh.read()
    docs = list(yaml.safe_load_all(content))
    if isinstance(docs[0], dict) and 'version' in docs[0]:
        versions = [d.get('version') for d in docs if isinstance(d, dict)]
        print(f'{f.name}: versions={versions}')
    elif isinstance(docs[0], dict) and 'templates' in docs[0]:
        templates = list(docs[0]['templates'].keys())
        print(f'{f.name}: version={docs[0].get(\"version\")}, sub-templates={templates}')
"

# Run prompt tests
python -m pytest tests/test_prompts_loader.py tests/test_prompt_versioning.py -v
```

Expected output:
```
agent_system.yaml: versions=['1.0']
faq_json_instruction.yaml: versions=['1.0']
faq_system.yaml: versions=['1.0']
guardrail.yaml: version=1.0, sub-templates=['pii_warning', 'off_topic_response', 'injection_warning', 'unsafe_content_response', 'default_refusal']
hitl_review.yaml: versions=['1.0']
rag_qa.yaml: versions=['1.0', '0.3', '0.2', '0.1']
```
