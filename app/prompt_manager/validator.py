"""Validation utilities for versioned prompt templates."""
import re
import logging
from typing import Dict, List, Optional

from app.prompt_manager.models import PromptDocument, PromptVersion

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a prompt template fails validation."""


def validate_prompt_document(doc: PromptDocument) -> List[str]:
    """Validate a PromptDocument and return a list of issues (empty = valid)."""
    issues: List[str] = []

    if not doc.name:
        issues.append("Prompt name is empty")

    if not doc.versions:
        issues.append("No versions defined")
        return issues

    for ver_id, version in doc.versions.items():
        version_issues = _validate_version(version, ver_id)
        issues.extend(version_issues)

    if doc.active_version and doc.active_version not in doc.versions:
        issues.append(f"Active version '{doc.active_version}' not found in versions")

    return issues


def _validate_version(version: PromptVersion, ver_id: str) -> List[str]:
    """Validate a single PromptVersion."""
    issues: List[str] = []
    prefix = f"Version '{ver_id}'"

    if not version.version:
        issues.append(f"{prefix}: version identifier is empty")

    if not version.template and not version.templates:
        issues.append(f"{prefix}: no template content defined")

    if version.template:
        _validate_template_variables(version.template, version.input_variables, issues, prefix)

    for sub_key, sub_template in version.templates.items():
        _validate_template_variables(sub_template, version.input_variables, issues, f"{prefix}.{sub_key}")

    return issues


def _validate_template_variables(template: str, declared_vars: List[str], issues: List[str], prefix: str):
    """Check that all declared variables are used and no undeclared variables appear."""
    # Find all {variable} patterns in the template
    found_vars = set(re.findall(r"\{(\w+)\}", template))

    declared_set = set(declared_vars)

    # Undeclared variables used in template
    undeclared = found_vars - declared_set
    for var in sorted(undeclared):
        issues.append(f"{prefix}: undeclared variable '{{{var}}}' used in template")

    # Declared variables not used in template
    unused = declared_set - found_vars
    if unused:
        issues.append(f"{prefix}: declared variables not used in template: {sorted(unused)}")


def get_template_text(doc: PromptDocument, sub_key: Optional[str] = None) -> str:
    """Get the active template text from a PromptDocument."""
    if not doc.active_version:
        raise ValidationError(f"No active version for prompt '{doc.name}'")

    version = doc.versions.get(doc.active_version)
    if not version:
        raise ValidationError(f"Active version '{doc.active_version}' not found for prompt '{doc.name}'")

    return version.get_template(sub_key)


def get_version(doc: PromptDocument, version_str: str) -> PromptVersion:
    """Get a specific version from a PromptDocument."""
    version = doc.versions.get(version_str)
    if not version:
        raise ValidationError(f"Version '{version_str}' not found for prompt '{doc.name}'")
    return version