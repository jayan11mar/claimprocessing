"""Pydantic models for versioned prompt templates."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChangelogEntry(BaseModel):
    version: str
    description: str


class PromptVersion(BaseModel):
    """A single version of a prompt template."""
    version: str
    author: str
    last_updated: str  # ISO date string
    changelog: Dict[str, str]
    model_compatibility: List[str] = Field(default_factory=lambda: ["gpt-4o-mini", "gpt-4"])
    input_variables: List[str] = Field(default_factory=list)
    template: str = ""
    templates: Dict[str, str] = Field(default_factory=dict)

    def get_template(self, sub_key: Optional[str] = None) -> str:
        """Return the template string, optionally by sub-key for multi-template files."""
        if sub_key:
            return self.templates.get(sub_key, "")
        return self.template


class PromptDocument(BaseModel):
    """Represents a loaded prompt file with its version history."""
    name: str
    versions: Dict[str, PromptVersion] = Field(default_factory=dict)
    active_version: str = ""
    activated_at: Optional[str] = None


class PromptSummary(BaseModel):
    """Summary of a prompt for listing endpoints."""
    name: str
    active_version: str
    available_versions: List[str]
    author: str


class VersionDetail(BaseModel):
    """Detail of a single version for history endpoints."""
    name: str
    version: str
    author: str
    last_updated: str
    changelog: Dict[str, str]
    model_compatibility: List[str]
    input_variables: List[str]
    activated_at: Optional[str] = None