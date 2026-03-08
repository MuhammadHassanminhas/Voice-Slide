"""
Tests for analytics.py — Phase 9.

Unit tests for filler word detection, word count, VADER sentiment scoring,
AnalyticsTracker class behaviour, and pipeline integration.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from analytics import (
    AnalyticsTracker, _count_fillers, _count_words, _score_sentiment,
    FILLER_LIST, _FILLER_RE,
)
import app as voice_app


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset app state for integration tests."""
    voice_app._last_nav_command_time = 0.0
    voice_app._speech_buffer.clear()
    voice_app._silence_counter = 0
    voice_app._is_speaking = False
    voice_app._current_slide_index = 0
    yield


def _fill_buffer():
    """Put dummy audio bytes into the speech buffer so _process_speech_buffer runs."""
    voice_app._speech_buffer.extend(b"\x00" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# Filler Word Detection Tests (1–12)
# ══════════════════════════════════════════════════════════════════════════════

def test_filler_um_detected():
    """'um' detected as a filler."""
    result = _count_fillers("I think um we should proceed")
    assert result["total"] == 1
    assert result["breakdown"]["um"] == 1


def test_filler_uh_detected():
    """'uh' detected as a filler."""
    result = _count_fillers("Revenue uh grew this quarter")
    assert result["total"] == 1
    assert result["breakdown"]["uh"] == 1


def test_filler_like_detected():
    """'like' detected as a filler."""
    result = _count_fillers("It was like really impressive")
    assert result["total"] == 1
    assert result["breakdown"]["like"] == 1


def test_filler_like_not_in_likelihood():
    """'like' inside 'likelihood' is NOT matched (word boundary)."""
    result = _count_fillers("The likelihood of success is high")
    assert result["total"] == 0


def test_filler_you_know_phrase():
    """Multi-word filler 'you know' matched as a phrase."""
    result = _count_fillers("We should you know focus more")
    assert result["total"] == 1
    assert result["breakdown"]["you know"] == 1


def test_filler_multiple_types():
    """Multiple different fillers in one segment."""
    result = _count_fillers("Um I think like you know it was basically fine")
    assert result["total"] == 4


def test_filler_repeated_same():
    """Same filler repeated multiple times."""
    result = _count_fillers("Um um um let me think")
    assert result["total"] == 3
    assert result["breakdown"]["um"] == 3


def test_filler_case_insensitive():
    """Fillers detected regardless of case."""
    result = _count_fillers("UM Like BASICALLY")
    assert result["total"] == 3


def test_filler_none_in_clean_speech():
    """Clean speech without fillers → total=0."""
    result = _count_fillers("Revenue grew eighteen percent year over year")
    assert result["total"] == 0


def test_filler_empty_string():
    """Empty string → total=0."""
    result = _count_fillers("")
    assert result["total"] == 0


def test_filler_sort_of_phrase():
    """Multi-word filler 'sort of' matched as a phrase."""
    result = _count_fillers("It was sort of expected")
    assert result["total"] == 1
    assert result["breakdown"]["sort of"] == 1


def test_filler_so_at_start():
    """'so' detected as a discourse marker."""
    result = _count_fillers("So we decided to expand")
    assert result["total"] == 1
    assert result["breakdown"]["so"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Word Count Tests (13–15)
# ══════════════════════════════════════════════════════════════════════════════

def test_word_count_normal():
    """Normal sentence word count."""
    assert _count_words("Revenue grew eighteen percent") == 4


def test_word_count_empty():
    """Empty string → 0 words."""
    assert _count_words("") == 0


def test_word_count_single_word():
    """Single word → 1."""
    assert _count_words("Hello") == 1


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment Scoring Tests (16–20)
# ══════════════════════════════════════════════════════════════════════════════

def test_sentiment_positive():
    """Clearly positive text → compound > 0.3."""
    result = _score_sentiment("This is a great achievement and we are very proud")
    assert result["compound"] > 0.3


def test_sentiment_negative():
    """Clearly negative text → compound < -0.3."""
    result = _score_sentiment("This is terrible and we failed badly")
    assert result["compound"] < -0.3


def test_sentiment_neutral():
    """Neutral/factual text → compound near 0."""
    result = _score_sentiment("The meeting is at three o'clock")
    assert -0.3 <= result["compound"] <= 0.3


def test_sentiment_returns_compound():
    """VADER dict contains 'compound' key."""
    result = _score_sentiment("Hello world")
    assert "compound" in result


def test_sentiment_empty_string():
    """Empty string → compound == 0.0."""
    result = _score_sentiment("")
    assert result["compound"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# AnalyticsTracker Tests (21–32)
# ══════════════════════════════════════════════════════════════════════════════

def test_tracker_initial_state():
    """Fresh tracker → all zeros, empty segments."""
    t = AnalyticsTracker()
    summary = t.get_summary()
    assert summary["total_words"] == 0
    assert summary["total_fillers"] == 0
    assert summary["segments"] == []


def test_tracker_record_one_segment():
    """Record one segment → correct word count and segments length."""
    t = AnalyticsTracker()
    t.record_segment("Hello everyone")
    summary = t.get_summary()
    assert summary["total_words"] == 2
    assert len(summary["segments"]) == 1


def test_tracker_record_multiple_segments():
    """Record 3 segments → correct aggregate word count."""
    t = AnalyticsTracker()
    t.record_segment("Hello everyone")          # 2 words
    t.record_segment("Revenue grew fast")       # 3 words
    t.record_segment("Next quarter outlook")    # 3 words
    summary = t.get_summary()
    assert summary["total_words"] == 8
    assert len(summary["segments"]) == 3


def test_tracker_filler_accumulation():
    """Fillers accumulate across multiple segments."""
    t = AnalyticsTracker()
    t.record_segment("Um like we should")       # um, like = 2
    t.record_segment("Uh basically yes")        # uh, basically = 2
    summary = t.get_summary()
    assert summary["total_fillers"] == 4


def test_tracker_wpm_calculation():
    """WPM computed from total words and elapsed time."""
    t = AnalyticsTracker()
    t.record_segment("One two three four five six seven eight nine ten")  # 10 words
    # Manually set session_start to 60 seconds ago for predictable WPM
    t._session_start = time.time() - 60.0
    summary = t.get_summary()
    # 10 words / 1 minute = 10 WPM
    assert 9.0 <= summary["avg_wpm"] <= 11.0


def test_tracker_filler_ratio():
    """Filler ratio = (fillers / words) * 100."""
    t = AnalyticsTracker()
    # 10 words total, 2 fillers (um, like)
    t.record_segment("Um one two three like four five six seven eight")
    summary = t.get_summary()
    assert summary["filler_ratio"] == 20.0


def test_tracker_reset_clears_all():
    """reset() returns tracker to initial state."""
    t = AnalyticsTracker()
    t.record_segment("Hello um world")
    t.reset()
    summary = t.get_summary()
    assert summary["total_words"] == 0
    assert summary["total_fillers"] == 0
    assert summary["segments"] == []


def test_tracker_segment_has_timestamp():
    """First segment timestamp is near 0.0."""
    t = AnalyticsTracker()
    t.record_segment("First utterance here")
    seg = t.get_summary()["segments"][0]
    assert seg["timestamp"] <= 0.1


def test_tracker_segment_has_sentiment():
    """Positive text → segment sentiment > 0."""
    t = AnalyticsTracker()
    t.record_segment("Great work everyone this is amazing")
    seg = t.get_summary()["segments"][0]
    assert seg["sentiment"] > 0


def test_tracker_filler_breakdown_aggregated():
    """filler_breakdown aggregates across all segments."""
    t = AnalyticsTracker()
    t.record_segment("Um we should um proceed")    # um=2
    t.record_segment("Like I think um we can")      # like=1, um=1
    summary = t.get_summary()
    assert summary["filler_breakdown"]["um"] == 3
    assert summary["filler_breakdown"]["like"] == 1


def test_tracker_empty_text_ignored():
    """record_segment('') does not add a segment."""
    t = AnalyticsTracker()
    t.record_segment("")
    t.record_segment("   ")
    summary = t.get_summary()
    assert len(summary["segments"]) == 0


def test_tracker_summary_structure():
    """get_summary() has all required top-level keys."""
    t = AnalyticsTracker()
    summary = t.get_summary()
    expected_keys = {
        "session_duration", "total_words", "total_fillers",
        "filler_ratio", "avg_wpm", "filler_breakdown", "segments",
    }
    assert set(summary.keys()) == expected_keys


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests (33–37)
# ══════════════════════════════════════════════════════════════════════════════

def test_analytics_route_returns_200():
    """GET /analytics → status 200."""
    client = voice_app.app.test_client()
    response = client.get("/analytics")
    assert response.status_code == 200


def test_api_analytics_returns_json():
    """GET /api/analytics → status 200 with expected JSON keys."""
    client = voice_app.app.test_client()
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_words" in data
    assert "total_fillers" in data
    assert "avg_wpm" in data
    assert "filler_breakdown" in data
    assert "segments" in data


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
@patch("app.analytics")
def test_pipeline_calls_record_segment(
    mock_analytics, mock_socketio, mock_transcribe, mock_classify,
    mock_fuzzy, mock_qa, mock_ctx, mock_vad
):
    """_process_speech_buffer() calls analytics.tracker.record_segment()."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "Hello um everyone"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = False

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_analytics.tracker.record_segment.assert_called_once_with("Hello um everyone")


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
@patch("app.analytics")
def test_pipeline_analytics_does_not_block_qa(
    mock_analytics, mock_socketio, mock_transcribe, mock_classify,
    mock_fuzzy, mock_qa, mock_ctx, mock_vad
):
    """Q&A detection still runs after analytics recording."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "What is the revenue"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = True
    mock_qa.search_notes.return_value = [
        {"slide_index": 1, "heading": "Revenue", "note_text": "Q3 680M", "score": 0.8},
    ]

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_analytics.tracker.record_segment.assert_called_once()
    mock_qa.is_question.assert_called_once_with("What is the revenue")
    mock_qa.search_notes.assert_called_once()


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
@patch("app.analytics")
def test_pipeline_analytics_does_not_block_intent(
    mock_analytics, mock_socketio, mock_transcribe, mock_classify,
    mock_fuzzy, mock_qa, mock_ctx, mock_vad
):
    """classify_intent() still runs after analytics recording."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "Next slide please"}
    mock_classify.return_value = {"intent": "NEXT_SLIDE", "confidence": 0.95}
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = False

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_analytics.tracker.record_segment.assert_called_once()
    mock_classify.assert_called_once_with("Next slide please")
