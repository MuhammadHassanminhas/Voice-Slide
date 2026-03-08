"""
VoiceSlide — Context Search (Phase 5)
Indexes slide text content and finds the best-matching slide for a natural
language query using sentence-transformer embeddings.

Query cleaning strips navigational noise before embedding.  A dual-index
(title + body) approach prevents semantic dilution from long slide text.

Uses the shared singleton from embeddings.py — no duplicate model loading.
"""

import logging
import re

from sentence_transformers import util
from embeddings import encode

logger = logging.getLogger("voiceslide.nlp.context_search")

# ── Module State ─────────────────────────────────────────────────────────────

_title_embeddings = None   # tensor (num_slides × 384), or None
_body_embeddings = None    # tensor (num_slides × 384), or None
_slide_titles: list[str] = []
_slide_bodies: list[str] = []
_slides: list[dict] = []   # raw slide dicts, used by keyword_highlighter
_num_slides: int = 0

SEARCH_THRESHOLD = 0.30


# ── Query Cleaning ──────────────────────────────────────────────────────────

_NAV_NOISE_RE = re.compile(
    r"\b(?:"
    r"(?:can you |could you |please |let's |let us |i want to |i'd like to )?"
    r"(?:move to|go to|go back to|jump to|skip to|switch to|take me to|"
    r"show me|find|where is|where was|where did we (?:discuss|talk about|cover))"
    r")\b(?:\s+the\b)?"
    r"|\b(?:the )?(?:slide|part|section|bit|stuff|thing|one)"
    r"(?:\s+(?:about|on|with|for|where))?\b",
    re.IGNORECASE,
)


def _clean_query(query: str) -> str:
    """Strip navigational noise from a voice query, keeping topic keywords."""
    cleaned = _NAV_NOISE_RE.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── Text Extraction ─────────────────────────────────────────────────────────

def _extract_title(slide: dict) -> str:
    """Extract the title/heading from a slide."""
    parts: list[str] = []
    for key in ("heading", "subheading"):
        val = slide.get(key)
        if val:
            parts.append(val)
    return " ".join(parts)


def _extract_body(slide: dict) -> str:
    """Extract body content (bullets, columns, captions, quotes, notes)."""
    parts: list[str] = []
    for key in ("body", "caption", "quote", "attribution"):
        val = slide.get(key)
        if val:
            parts.append(val)
    for item in slide.get("items", []):
        parts.append(item)
    for side in ("left", "right"):
        col = slide.get(side)
        if isinstance(col, dict):
            if col.get("title"):
                parts.append(col["title"])
            for item in col.get("items", []):
                parts.append(item)
    if slide.get("notes"):
        parts.append(slide["notes"])
    return " ".join(parts)


# ── Public API ───────────────────────────────────────────────────────────────

def build_index(slides: list[dict]) -> None:
    """Extract text from each slide and compute dual embeddings (title + body).

    Safe to call multiple times — each call replaces the previous index.
    """
    global _title_embeddings, _body_embeddings, _slide_titles, _slide_bodies, _slides, _num_slides

    _slides = list(slides)
    _slide_titles = [_extract_title(s) for s in slides]
    _slide_bodies = [_extract_body(s) for s in slides]
    _num_slides = len(slides)

    if _num_slides > 0:
        title_texts = [t if t.strip() else " " for t in _slide_titles]
        _title_embeddings = encode(title_texts)

        body_texts = [b if b.strip() else " " for b in _slide_bodies]
        _body_embeddings = encode(body_texts)

        logger.info("Context index built: %d slides (title + body).", _num_slides)
    else:
        _title_embeddings = None
        _body_embeddings = None
        logger.warning("No slides to index — context search will be empty.")


def search(query: str, threshold: float = SEARCH_THRESHOLD) -> dict | None:
    """Find the slide whose content best matches *query*.

    Cleans navigational noise from the query, then searches both the title
    and body embedding indices.  Returns the best match across both channels.

    Returns:
        dict with keys ``slide_index`` (0-based), ``score``, ``matched_text``
        if a match is found above *threshold*; ``None`` otherwise.
    """
    if _title_embeddings is None or _num_slides == 0:
        return None

    cleaned = _clean_query(query)
    search_query = cleaned if cleaned else query

    query_embedding = encode([search_query])

    title_hits = util.semantic_search(query_embedding, _title_embeddings, top_k=1)
    body_hits = util.semantic_search(query_embedding, _body_embeddings, top_k=1)

    best_title = title_hits[0][0] if title_hits and title_hits[0] else None
    best_body = body_hits[0][0] if body_hits and body_hits[0] else None

    candidates = []
    if best_title:
        candidates.append(best_title)
    if best_body:
        candidates.append(best_body)

    if not candidates:
        return None

    best = max(candidates, key=lambda h: h["score"])

    if best["score"] >= threshold:
        idx = best["corpus_id"]
        matched_text = _slide_titles[idx] or _slide_bodies[idx]
        logger.info(
            "Context search: '%.60s' → slide %d (score: %.2f)",
            search_query, idx, best["score"],
        )
        return {
            "slide_index": idx,
            "score": float(best["score"]),
            "matched_text": matched_text[:100],
        }

    logger.debug("Context search: no match above %.2f for '%.60s'", threshold, search_query)
    return None
