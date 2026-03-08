"""
VoiceSlide — Intent Classifier
Three-tier matching:
  0. Heuristics  — length penalty + end-of-sentence bias (< 0.1 ms).
  1. Fast-path   — O(1) dict lookup for exact canonical phrases (< 1 ms).
  2. Slow-path   — cosine similarity on sentence embeddings (~ 10-20 ms).
Includes slide-number extraction for GOTO_SLIDE commands (digits, cardinals, ordinals).
"""

import re
import logging

from sentence_transformers import util
from embeddings import encode

logger = logging.getLogger("voiceslide.nlp.intent")


# ── Canonical Phrases (used by both fast-path and embedding lookup) ──────────

CANONICAL_INTENTS = {
    "NEXT_SLIDE": [
        "next slide", "next slide please", "move forward",
        "skip ahead", "go to the next one", "continue"
    ],
    "PREV_SLIDE": [
        "previous slide", "go back", "last slide",
        "let's go back", "go back a slide"
    ],
    "NEXT_POINT": [
        "next point", "next bullet", "show next item",
        "next item please"
    ],
    "PREV_POINT": [
        "previous point", "go back one point",
        "previous bullet", "last point"
    ],
    "GOTO_SLIDE": [
        "go to slide", "jump to slide", "switch to slide",
        "skip to slide", "show slide"
    ],
    "START_PRESENTATION": [
        "start presentation", "begin the talk", "let's start"
    ],
    "END_PRESENTATION": [
        "end presentation", "stop presenting", "that's all", "we are done here"
    ],
}

# Threshold required to trigger an intent via embedding similarity
CONFIDENCE_THRESHOLD = 0.55


# ── Fast-Path Exact Matcher ──────────────────────────────────────────────────

FAST_PATH_MAP: dict[str, str] = {}
for _intent, _phrases in CANONICAL_INTENTS.items():
    for _phrase in _phrases:
        FAST_PATH_MAP[_phrase] = _intent


def _fast_path_match(text: str) -> dict | None:
    """
    O(1) exact-match, then O(n) end-of-sentence scan.
    Returns None if no match.
    """
    normalized = text.strip().lower()

    # ── 1. Exact whole-string match (confidence 1.0) ─────────────────
    intent = FAST_PATH_MAP.get(normalized)
    if intent:
        return {"intent": intent, "confidence": 1.0}

    # ── 2. End-of-sentence match (confidence 0.95) ───────────────────
    #    Text ENDS WITH a canonical phrase, preceded by a word boundary.
    for phrase, intent in FAST_PATH_MAP.items():
        if normalized.endswith(phrase):
            prefix = normalized[: -len(phrase)]
            if prefix == "" or prefix.endswith(" "):
                return {"intent": intent, "confidence": 0.95}

    return None


# ── Slide Number Extraction ──────────────────────────────────────────────────

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

ORDINAL_TO_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
}


def _extract_slide_number(text: str) -> int | None:
    """Extract a slide number from text.  Priority: digit → cardinal → ordinal."""
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text)
    if match:
        return int(match.group(1))

    lower = text.lower()

    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            return num

    for word, num in ORDINAL_TO_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            return num

    return None


def _is_mid_sentence_false_positive(text: str, intent: str) -> bool:
    """Return True if a canonical phrase sits mid-sentence (2+ trailing words)."""
    if intent == "GOTO_SLIDE":
        return False
    normalized = text.strip().lower()
    min_trailing: int | None = None
    for phrase in CANONICAL_INTENTS.get(intent, []):
        idx = normalized.find(phrase)
        if idx == -1:
            continue
        if idx > 0 and normalized[idx - 1] != " ":
            continue
        after = normalized[idx + len(phrase):].strip()
        trailing = len(after.split()) if after else 0
        if min_trailing is None or trailing < min_trailing:
            min_trailing = trailing
    return min_trailing is not None and min_trailing >= 2


# ── Embedding-Based Classifier (Slow Path) ──────────────────────────────────

_intent_embeddings = {}
_intent_mapping: dict[int, str] = {}


def _initialize_classifier():
    """Pre-computes embeddings for all canonical phrases."""
    global _intent_embeddings, _intent_mapping
    if len(_intent_embeddings) > 0:
        return  # already initialized

    logger.info("Initializing Intent Classifier canonical embeddings...")

    all_phrases: list[str] = []
    for intent, phrases in CANONICAL_INTENTS.items():
        for phrase in phrases:
            all_phrases.append(phrase)
            _intent_mapping[len(all_phrases) - 1] = intent

    _intent_embeddings = encode(all_phrases)
    logger.info("Intent Classifier initialized with %d canonical phrases.", len(all_phrases))


# ── Public API ───────────────────────────────────────────────────────────────

def classify_intent(text: str) -> dict:
    """
    Classifies a raw text string into a structured intent.

    Returns:
        dict: {"intent": str, "confidence": float}
              For GOTO_SLIDE also includes "slide_number": int | None
    """
    if not text or not text.strip():
        return {"intent": "NONE", "confidence": 0.0}

    # ── Tier 0a: Length Penalty ──────────────────────────────────────────
    words = text.split()
    if len(words) > 10:
        logger.debug("Length penalty: '%s' has %d words → NONE", text, len(words))
        return {"intent": "NONE", "confidence": 0.0}

    # ── Tier 1: fast-path exact match ────────────────────────────────────
    fast = _fast_path_match(text)
    if fast:
        if fast["intent"] == "GOTO_SLIDE":
            fast["slide_number"] = _extract_slide_number(text)
        logger.info("Fast-path matched '%s' → %s", text, fast["intent"])
        return fast

    # ── Tier 2: embedding cosine similarity ──────────────────────────────
    _initialize_classifier()

    query_embedding = encode([text])
    hits = util.semantic_search(query_embedding, _intent_embeddings, top_k=1)

    if hits and hits[0]:
        best_hit = hits[0][0]
        score = best_hit["score"]

        if score >= CONFIDENCE_THRESHOLD:
            matched_intent = _intent_mapping[best_hit["corpus_id"]]

            # ── Tier 0c: reject mid-sentence false positives ─────────
            if _is_mid_sentence_false_positive(text, matched_intent):
                logger.debug("Mid-sentence false positive rejected: '%s'", text)
                return {"intent": "NONE", "confidence": 0.0}

            result = {
                "intent": matched_intent,
                "confidence": float(score),
            }
            if matched_intent == "GOTO_SLIDE":
                result["slide_number"] = _extract_slide_number(text)
            logger.info("Embedding matched '%s' → %s (score: %.2f)", text, matched_intent, score)
            return result

    return {"intent": "NONE", "confidence": 0.0}
