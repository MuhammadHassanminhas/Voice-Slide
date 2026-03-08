"""
VoiceSlide — Q&A Assistant (Phase 8)
Detects spoken questions from live transcripts and searches slide notes
using the shared sentence-transformer embedding model for semantic similarity.

Uses the singleton from embeddings.py — no duplicate model loading.
"""

import logging
import re

from sentence_transformers import util
from embeddings import encode
import context_search

logger = logging.getLogger("voiceslide.nlp.qa_assistant")

# ── Module State ─────────────────────────────────────────────────────────────

_note_entries: list[dict] = []
_note_embeddings = None  # Tensor or None

NOTES_SEARCH_THRESHOLD = 0.25


# ── Question Detection ──────────────────────────────────────────────────────

_QUESTION_RE = re.compile(
    r"^[\s\-\u2013\u2014\u2022\u00b7\u201c\u201d\"\'\(\[,;:\u2026.]*(?:"
    r"(?:what|which|where|when|who|whom|whose|why|how)"
    r"|(?:is|are|was|were|do|does|did|can|could|will|would|shall"
    r"|should|may|might|has|have|had)"
    r")\b",
    re.IGNORECASE,
)

_TRAILING_QUESTION_RE = re.compile(r"\?\s*$")


def is_question(text: str) -> bool:
    """Detect if a transcript segment is a question.

    Uses a two-tier heuristic:
    1. Check if the utterance starts with a WH-word or auxiliary verb
       (interrogative word order).
    2. Check for a trailing '?' (unreliable from STT but free to check).

    Short utterances (< 3 words) are excluded to avoid false positives
    from filler words like "what" or "how".
    """
    stripped = text.strip()
    words = stripped.split()
    if len(words) < 3:
        return False

    if _TRAILING_QUESTION_RE.search(stripped):
        return True

    return bool(_QUESTION_RE.match(stripped))


# ── Notes Extraction & Indexing ─────────────────────────────────────────────

def _extract_notes() -> list[dict]:
    """Extract notes from the slides already loaded in context_search._slides."""
    entries = []
    for idx, slide in enumerate(context_search._slides):
        note = slide.get("notes", "").strip()
        if note:
            heading = slide.get("heading", f"Slide {idx + 1}")
            entries.append({
                "slide_index": idx,
                "heading": heading,
                "note_text": note,
            })
    return entries


def build_notes_index() -> None:
    """Embed all non-empty slide notes for semantic search.

    Reads from context_search._slides (already in memory).
    Safe to call multiple times — each call replaces the previous index.
    """
    global _note_entries, _note_embeddings

    _note_entries = _extract_notes()

    if _note_entries:
        texts = [e["note_text"] for e in _note_entries]
        _note_embeddings = encode(texts)
        logger.info("Q&A notes index built: %d notes embedded.", len(_note_entries))
    else:
        _note_embeddings = None
        logger.info("Q&A notes index: no notes found in slides.")


# ── Public API ───────────────────────────────────────────────────────────────

def search_notes(question: str, top_k: int = 3) -> list[dict]:
    """Find the top-k slide notes most relevant to the question.

    Returns a list of dicts:
        [{"slide_index": int, "heading": str, "note_text": str, "score": float}, ...]

    Returns an empty list if no notes are indexed or no match exceeds threshold.
    """
    if _note_embeddings is None or not _note_entries:
        return []

    query_embedding = encode([question])
    hits = util.semantic_search(query_embedding, _note_embeddings, top_k=top_k)

    results = []
    for hit in hits[0]:
        if hit["score"] >= NOTES_SEARCH_THRESHOLD:
            entry = _note_entries[hit["corpus_id"]]
            results.append({
                "slide_index": entry["slide_index"],
                "heading": entry["heading"],
                "note_text": entry["note_text"],
                "score": round(float(hit["score"]), 3),
            })

    return results
