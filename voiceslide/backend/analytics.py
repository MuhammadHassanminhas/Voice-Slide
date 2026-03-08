"""
VoiceSlide — Speech Analytics Tracker (Phase 9)
Silently records filler words, speaking pace, and sentiment for each
transcript segment during a live presentation session.
"""

import logging
import re
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("voiceslide.analytics")

# VADER is a lexicon lookup — instantiate once at import time (~1ms, ~300 KB)
_analyzer = SentimentIntensityAnalyzer()


# ── Filler Word Detection ───────────────────────────────────────────────────

FILLER_LIST = [
    "um", "uh", "uhm", "umm",
    "like",
    "you know",
    "i mean",
    "sort of", "kind of",
    "basically", "actually", "literally",
    "right", "okay", "so",
]

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in sorted(FILLER_LIST, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _count_fillers(text: str) -> dict:
    """Count filler word occurrences in a text segment.

    Returns:
        {"total": int, "breakdown": {"um": 2, "like": 1, ...}}
    """
    matches = _FILLER_RE.findall(text.lower())
    breakdown = {}
    for m in matches:
        key = m.lower()
        breakdown[key] = breakdown.get(key, 0) + 1
    return {"total": len(matches), "breakdown": breakdown}


# ── Word Count ──────────────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    """Count words in a text segment (simple whitespace split)."""
    return len(text.split())


# ── Sentiment Scoring ───────────────────────────────────────────────────────

def _score_sentiment(text: str) -> dict:
    """Score sentiment of a text segment using VADER.

    Returns the full VADER dict:
        {"neg": float, "neu": float, "pos": float, "compound": float}
    """
    return _analyzer.polarity_scores(text)


# ── Analytics Tracker Class ─────────────────────────────────────────────────

class AnalyticsTracker:
    """Tracks speech analytics for a single presentation session."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all metrics for a new session."""
        self._segments = []
        self._session_start = None
        self._total_words = 0
        self._total_fillers = 0

    def record_segment(self, text: str) -> None:
        """Record metrics for one transcript segment.

        Called once per utterance from _process_speech_buffer().
        """
        if not text or not text.strip():
            return

        now = time.time()
        if self._session_start is None:
            self._session_start = now

        word_count = _count_words(text)
        filler_result = _count_fillers(text)
        sentiment = _score_sentiment(text)

        self._total_words += word_count
        self._total_fillers += filler_result["total"]

        self._segments.append({
            "timestamp": round(now - self._session_start, 2),
            "text": text,
            "word_count": word_count,
            "filler_count": filler_result["total"],
            "fillers_found": filler_result["breakdown"],
            "sentiment": sentiment["compound"],
        })

    def get_summary(self) -> dict:
        """Return the full analytics payload for the dashboard."""
        if not self._segments:
            return {
                "session_duration": 0.0,
                "total_words": 0,
                "total_fillers": 0,
                "filler_ratio": 0.0,
                "avg_wpm": 0.0,
                "filler_breakdown": {},
                "segments": [],
            }

        elapsed = time.time() - self._session_start
        elapsed_min = elapsed / 60.0

        agg_breakdown = {}
        for seg in self._segments:
            for filler, count in seg["fillers_found"].items():
                agg_breakdown[filler] = agg_breakdown.get(filler, 0) + count

        return {
            "session_duration": round(elapsed, 2),
            "total_words": self._total_words,
            "total_fillers": self._total_fillers,
            "filler_ratio": round(
                (self._total_fillers / self._total_words * 100) if self._total_words > 0 else 0.0, 2
            ),
            "avg_wpm": round(self._total_words / elapsed_min if elapsed_min > 0 else 0.0, 1),
            "filler_breakdown": agg_breakdown,
            "segments": [
                {
                    "timestamp": s["timestamp"],
                    "text": s["text"],
                    "word_count": s["word_count"],
                    "filler_count": s["filler_count"],
                    "sentiment": s["sentiment"],
                }
                for s in self._segments
            ],
        }


# Module-level singleton
tracker = AnalyticsTracker()
