"""Central registry for versioned prompt templates with startup load, hot-reload, and rollback."""
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import get_settings
from app.prompt_manager.loader import load_all_prompts, start_watcher
from app.prompt_manager.models import PromptDocument, PromptVersion
from app.prompt_manager.validator import validate_prompt_document, get_template_text, ValidationError

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Thread-safe registry for versioned prompts with active-version pointer."""

    def __init__(self):
        self._lock = threading.RLock()
        self._prompts: Dict[str, PromptDocument] = {}
        self._observer = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self, prompts_dir: Optional[Path] = None) -> int:
        """Load all prompts from the YAML directory. Returns count of loaded prompts."""
        settings = get_settings()
        if prompts_dir is None:
            prompts_dir = Path(settings.PROMPTS_DIR)
            if not prompts_dir.is_absolute():
                prompts_dir = Path(__file__).resolve().parent.parent.parent / prompts_dir

        with self._lock:
            self._prompts = load_all_prompts(prompts_dir)
            # Validate all loaded prompts
            for name, doc in list(self._prompts.items()):
                issues = validate_prompt_document(doc)
                if issues:
                    logger.warning("Validation issues for prompt '%s': %s", name, issues)

            count = len(self._prompts)
            logger.info("PromptRegistry loaded %d prompt(s)", count)
            return count

    def start_hot_reload(self, prompts_dir: Optional[Path] = None) -> None:
        """Start watchdog-based hot-reload on the prompts directory."""
        settings = get_settings()
        if prompts_dir is None:
            prompts_dir = Path(settings.PROMPTS_DIR)
            if not prompts_dir.is_absolute():
                prompts_dir = Path(__file__).resolve().parent.parent.parent / prompts_dir

        self._observer = start_watcher(self._prompts, prompts_dir)

    def stop_hot_reload(self) -> None:
        """Stop the watchdog observer."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    # ── Accessors ──────────────────────────────────────────────────────────

    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary of all registered prompts."""
        with self._lock:
            result = {}
            for name, doc in self._prompts.items():
                result[name] = {
                    "name": doc.name,
                    "active_version": doc.active_version,
                    "available_versions": sorted(doc.versions.keys(), key=lambda v: [int(x) for x in v.split(".")]),
                }
            return result

    def get_prompt(self, name: str) -> Optional[PromptDocument]:
        """Get a prompt document by name."""
        with self._lock:
            return self._prompts.get(name)

    def get_template(self, name: str, sub_key: Optional[str] = None) -> str:
        """Get the active template text for a named prompt."""
        with self._lock:
            doc = self._prompts.get(name)
            if not doc:
                raise ValidationError(f"Prompt '{name}' not found in registry")
            return get_template_text(doc, sub_key)

    def get_version_history(self, name: str) -> list:
        """Get version history for a prompt (ordered ascending)."""
        with self._lock:
            doc = self._prompts.get(name)
            if not doc:
                return []
            versions = []
            for ver_id in sorted(doc.versions.keys(), key=lambda v: [int(x) for x in v.split(".")]):
                version = doc.versions[ver_id]
                versions.append({
                    "name": doc.name,
                    "version": version.version,
                    "author": version.author,
                    "last_updated": version.last_updated,
                    "changelog": version.changelog,
                    "model_compatibility": version.model_compatibility,
                    "input_variables": version.input_variables,
                    "activated_at": doc.activated_at if doc.active_version == ver_id else None,
                })
            return versions

    def activate_version(self, name: str, version_str: str) -> bool:
        """Activate a specific version of a prompt. Returns True if successful."""
        with self._lock:
            doc = self._prompts.get(name)
            if not doc:
                logger.warning("Cannot activate: prompt '%s' not found", name)
                return False
            if version_str not in doc.versions:
                logger.warning("Cannot activate: version '%s' not found for prompt '%s'", version_str, name)
                return False

            doc.active_version = version_str
            doc.activated_at = datetime.utcnow().isoformat() + "Z"
            logger.info("Activated version '%s' for prompt '%s'", version_str, name)
            return True

    def get_active_version(self, name: str) -> Optional[str]:
        """Get the currently active version string for a prompt."""
        with self._lock:
            doc = self._prompts.get(name)
            if not doc:
                return None
            return doc.active_version


# Singleton global registry
_registry: Optional[PromptRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> PromptRegistry:
    """Get or create the singleton PromptRegistry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PromptRegistry()
    return _registry


def initialize_prompts(prompts_dir: Optional[Path] = None) -> PromptRegistry:
    """Initialize the prompt registry: load prompts and start hot-reload."""
    registry = get_registry()
    registry.load(prompts_dir)
    if get_settings().ENABLE_PROMPT_MANAGER:
        registry.start_hot_reload(prompts_dir)
    return registry