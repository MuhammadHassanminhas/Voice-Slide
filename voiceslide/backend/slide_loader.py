"""
VoiceSlide — Slide Loader
Owns loading, validating, saving, and extracting text from slides.json.
The Backend Agent calls these functions — no Flask routes here.
"""

import json
import logging
import os
import shutil
from typing import Any

from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

# ── JSON Schema Definition ──────────────────────────────────────────────────

SLIDE_SCHEMA = {
    "type": "object",
    "required": ["version", "title", "slides"],
    "properties": {
        "version": {"type": "string"},
        "title": {"type": "string"},
        "author": {"type": "string"},
        "theme": {"type": "string", "enum": ["dark", "light"]},
        "created_at": {"type": "string"},
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": ["title", "bullets", "image", "text", "two-column", "quote"],
                    },
                    "heading": {"type": "string"},
                    "subheading": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                    "body": {"type": "string"},
                    "image_url": {"type": "string"},
                    "caption": {"type": "string"},
                    "quote": {"type": "string"},
                    "attribution": {"type": "string"},
                    "left": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "right": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "notes": {"type": "string"},
                },
            },
        },
    },
}

_validator = Draft7Validator(SLIDE_SCHEMA)


# ── Public API ───────────────────────────────────────────────────────────────


def validate_schema(data: dict) -> tuple[bool, list[str]]:
    """Validate slide data against the canonical schema.

    Args:
        data: A dict representing the slides.json content.

    Returns:
        A tuple of (is_valid, list_of_error_messages).

    Example:
        >>> ok, errors = validate_schema({"version": "1.0", "title": "T", "slides": []})
        >>> ok
        True
    """
    errors = sorted(_validator.iter_errors(data), key=lambda e: list(e.path))
    messages = [f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in errors]
    return (len(messages) == 0, messages)


def load_slides(path: str) -> dict:
    """Load and validate slides.json from *path*.

    Args:
        path: Absolute path to a slides.json file.

    Returns:
        The parsed and validated slide data dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or fails schema validation.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Slide file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    is_valid, errors = validate_schema(data)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    logger.info("Loaded %d slides from %s", len(data.get("slides", [])), path)
    return data


def save_slides(data: dict, path: str) -> bool:
    """Validate and persist slide data to disk, creating a backup first.

    Args:
        data: A dict matching the slides.json schema.
        path: Absolute path to write to.

    Returns:
        True on success.

    Raises:
        ValueError: If validation fails.
    """
    is_valid, errors = validate_schema(data)
    if not is_valid:
        raise ValueError(f"Cannot save — schema validation failed: {'; '.join(errors)}")

    # Backup existing file
    if os.path.isfile(path):
        backup = path.replace(".json", "_backup.json")
        shutil.copy2(path, backup)
        logger.info("Backed up existing slides to %s", backup)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d slides to %s", len(data.get("slides", [])), path)
    return True


def get_slide_texts(data: dict) -> list[dict[str, Any]]:
    """Extract all text content from slides for NLP processing.

    Returns a flat list of dicts with ``slide_index`` and ``texts`` (list of
    lowercase strings suitable for fuzzy matching).

    Args:
        data: A validated slides.json dict.

    Returns:
        List of ``{"slide_index": int, "texts": list[str]}``.

    Example:
        >>> texts = get_slide_texts(data)
        >>> texts[0]
        {"slide_index": 0, "texts": ["q3 business review", ...]}
    """
    results: list[dict[str, Any]] = []
    for idx, slide in enumerate(data.get("slides", [])):
        texts: list[str] = []

        if slide.get("heading"):
            texts.append(slide["heading"].lower().strip())
        if slide.get("subheading"):
            texts.append(slide["subheading"].lower().strip())
        if slide.get("items"):
            texts.extend(item.lower().strip() for item in slide["items"])
        if slide.get("body"):
            texts.append(slide["body"].lower().strip())
        if slide.get("quote"):
            texts.append(slide["quote"].lower().strip())
        if slide.get("caption"):
            texts.append(slide["caption"].lower().strip())

        # Two-column content
        for col in ("left", "right"):
            column = slide.get(col)
            if column and isinstance(column, dict):
                if column.get("title"):
                    texts.append(column["title"].lower().strip())
                if column.get("items"):
                    texts.extend(item.lower().strip() for item in column["items"])

        results.append({"slide_index": idx, "texts": texts})

    return results
