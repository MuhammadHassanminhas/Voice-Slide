"""
VoiceSlide — Keyword Highlighter (Phase 6)
Fuzzy-matches a speech transcript against the current slide's text spans.
Used as an interceptor to block the Universal Fallback when the presenter
is simply reading the active slide aloud.

Uses thefuzz (partial_ratio) for substring-aware fuzzy matching, and a
regex-based emphasis detector for verbal stress phrases.
"""

import logging
import re

from thefuzz import fuzz
import context_search

logger = logging.getLogger("voiceslide.nlp.keyword_highlighter")

HIGHLIGHT_THRESHOLD = 65  # partial_ratio score (0–100)


# ── Emphasis Detection ──────────────────────────────────────────────────────

_EMPHASIS_RE = re.compile(
    r"\b(?:(?:this\s+is\s+)?(?:very\s+|really\s+|extremely\s+|super\s+)?important"
    r"|key\s+point|remember\s+this|pay\s+attention(?:\s+to\s+this)?"
    r"|note\s+this|critical|crucial|essential)\b",
    re.IGNORECASE,
)


def _detect_emphasis(transcript: str) -> bool:
    """Return True if the transcript contains emphasis trigger phrases."""
    return bool(_EMPHASIS_RE.search(transcript))


# ── Slide Text Span Extraction ──────────────────────────────────────────────

def _get_slide_text_spans(slide: dict) -> list[str]:
    """Return a list of individual text spans from a slide.

    Each span is a matchable unit (heading, bullet item, caption, quote, etc.).
    """
    spans = []
    if slide.get("heading"):
        spans.append(slide["heading"])
    if slide.get("subheading"):
        spans.append(slide["subheading"])
    for item in slide.get("items", []):
        spans.append(item)
    if slide.get("caption"):
        spans.append(slide["caption"])
    if slide.get("quote"):
        spans.append(slide["quote"])
    if slide.get("attribution"):
        spans.append(slide["attribution"])
    for side in ("left", "right"):
        col = slide.get(side, {})
        if isinstance(col, dict):
            if col.get("title"):
                spans.append(col["title"])
            for item in col.get("items", []):
                spans.append(item)
    return spans


# ── Public API ──────────────────────────────────────────────────────────────

def fuzzy_match_current_slide(transcript: str, slide_index: int) -> dict | None:
    """Check if the transcript fuzzy-matches any text on the slide at *slide_index*.

    Returns a highlight payload dict if matched, or None if no match.
    """
    if not context_search._slides or slide_index >= len(context_search._slides):
        return None

    words = transcript.strip().split()
    if len(words) < 3:
        return None

    slide = context_search._slides[slide_index]
    spans = _get_slide_text_spans(slide)

    if not spans:
        return None

    transcript_lower = transcript.lower().strip()
    best_score = 0
    best_span = ""

    for span in spans:
        span_lower = span.lower().strip()
        score = fuzz.partial_ratio(transcript_lower, span_lower)
        if score > best_score:
            best_score = score
            best_span = span

    if best_score >= HIGHLIGHT_THRESHOLD:
        return {
            "slide_index": slide_index,
            "matched_span": best_span,
            "score": best_score,
            "emphasis": _detect_emphasis(transcript),
        }

    return None
