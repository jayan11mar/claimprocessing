"""Load versioned YAML prompt files with metadata."""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

from app.prompt_manager.models import PromptDocument, PromptVersion

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PromptLoadError(Exception):
    """Raised when a prompt file cannot be loaded or parsed."""


def _parse_yaml_documents(filepath: Path) -> List[dict]:
    """Parse all YAML documents from a file (supports multi-document YAML with --- separator)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    documents = list(yaml.safe_load_all(content))
    return [doc for doc in documents if doc is not None]


def _build_version(doc: dict, filepath: Path) -> Optional[PromptVersion]:
    """Build a PromptVersion from a parsed YAML document dict."""
    try:
        version = doc.get("version", "1.0")
        author = doc.get("author", "unknown")
        last_updated = doc.get("last_updated", "unknown")
        changelog = doc.get("changelog", {})
        model_compatibility = doc.get("model_compatibility", ["gpt-4o-mini", "gpt-4"])
        input_variables = doc.get("input_variables", [])
        template = doc.get("template", "")
        templates = doc.get("templates", {})

        return PromptVersion(
            version=str(version),
            author=str(author),
            last_updated=str(last_updated),
            changelog=changelog,
            model_compatibility=model_compatibility,
            input_variables=input_variables,
            template=template,
            templates=templates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse prompt version in %s: %s", filepath.name, exc)
        return None


def load_prompt_file(filepath: Path) -> Optional[PromptDocument]:
    """Load a single YAML prompt file and return a PromptDocument with all versions."""
    if not filepath.exists():
        logger.warning("Prompt file not found: %s", filepath)
        return None

    try:
        docs = _parse_yaml_documents(filepath)
    except (yaml.YAMLError, IOError) as exc:
        logger.error("Failed to parse YAML file %s: %s", filepath, exc)
        return None

    if not docs:
        logger.warning("Empty YAML file: %s", filepath)
        return None

    name = filepath.stem  # e.g. "faq_system", "rag_qa"
    versions: Dict[str, PromptVersion] = {}
    first_version = None

    for doc in docs:
        version_obj = _build_version(doc, filepath)
        if version_obj:
            ver_id = version_obj.version
            if ver_id in versions:
                ver_id = f"{ver_id}_{len(versions)}"  # deduplicate
            versions[ver_id] = version_obj
            if first_version is None:
                first_version = ver_id

    if not versions:
        logger.warning("No valid versions found in %s", filepath)
        return None

    # Activate the highest version by default
    sorted_versions = sorted(versions.keys(), key=lambda v: [int(x) for x in v.split(".")])
    active = sorted_versions[-1]

    doc = PromptDocument(
        name=name,
        versions=versions,
        active_version=active,
    )
    logger.info("Loaded prompt '%s' with %d version(s), active=%s", name, len(versions), active)
    return doc


def load_all_prompts(prompts_dir: Optional[Path] = None) -> Dict[str, PromptDocument]:
    """Load all YAML prompt files from the prompts directory."""
    prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR
    if not prompts_dir.exists():
        logger.warning("Prompts directory not found: %s", prompts_dir)
        return {}

    registry: Dict[str, PromptDocument] = {}
    for filepath in sorted(prompts_dir.glob("*.yaml")):
        doc = load_prompt_file(filepath)
        if doc:
            registry[doc.name] = doc
    return registry


class PromptFileHandler(FileSystemEventHandler):
    """Watchdog handler that reloads prompts on file changes."""

    def __init__(self, registry_ref: dict):
        self._registry_ref = registry_ref
        self._lock = threading.Lock()

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".yaml"):
            return

        filepath = Path(event.src_path)
        self._reload_file(filepath)

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".yaml"):
            return
        filepath = Path(event.src_path)
        self._reload_file(filepath)

    def _reload_file(self, filepath: Path):
        doc = load_prompt_file(filepath)
        if doc is None:
            return
        with self._lock:
            self._registry_ref[doc.name] = doc
        logger.info("Hot-reloaded prompt '%s' from %s", doc.name, filepath.name)


def start_watcher(registry: dict, prompts_dir: Optional[Path] = None) -> Observer:
    """Start a watchdog observer on the prompts directory for hot-reload."""
    prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR
    if not prompts_dir.exists():
        logger.warning("Cannot start watcher, directory not found: %s", prompts_dir)
        return None

    event_handler = PromptFileHandler(registry)
    observer = Observer()
    observer.schedule(event_handler, str(prompts_dir), recursive=False)
    observer.start()
    logger.info("Prompt watcher started on %s", prompts_dir)
    return observer