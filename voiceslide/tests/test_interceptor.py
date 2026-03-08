"""
Tests for the Phase 6 Interceptor Pattern — Fuzzy Match blocks Universal Fallback.

Integration tests verify the interceptor path in _process_speech_buffer():
when the intent classifier returns NONE and the transcript fuzzy-matches the
current slide, highlight_text is emitted and context_search.search is NOT called.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import app as voice_app


@pytest.fixture(autouse=True)
def reset_state():
    """Reset nav cooldown, speech buffer, and slide index before each test."""
    voice_app._last_nav_command_time = 0.0
    voice_app._speech_buffer.clear()
    voice_app._silence_counter = 0
    voice_app._is_speaking = False
    voice_app._current_slide_index = 0
    yield


def _fill_buffer():
    """Put dummy audio bytes into the speech buffer so _process_speech_buffer runs."""
    voice_app._speech_buffer.extend(b"\x00" * 100)


# ── Interceptor Tests ───────────────────────────────────────────────────────

@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_interceptor_blocks_fallback(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """Classifier → NONE, fuzzy match → hit → highlight_text emitted, search NOT called."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "physically separate storage for instructions"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = {
        "slide_index": 1,
        "matched_span": "Physically separate storage for instructions and data",
        "score": 88,
        "emphasis": False,
    }

    voice_app._current_slide_index = 1
    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_socketio.emit.assert_any_call("highlight_text", {
        "slide_index": 1,
        "matched_span": "Physically separate storage for instructions and data",
        "score": 88,
        "emphasis": False,
    })
    mock_ctx.search.assert_not_called()


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_interceptor_miss_falls_through(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """Classifier → NONE, fuzzy match → None → context_search.search IS called."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "quantum physics dark matter"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_ctx.search.assert_called_once()
    highlight_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "highlight_text"]
    assert len(highlight_calls) == 0


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_command_bypasses_interceptor(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """Classifier → NEXT_SLIDE → command emitted, fuzzy match NOT called."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "next slide"}
    mock_classify.return_value = {"intent": "NEXT_SLIDE", "confidence": 1.0}
    mock_ctx._num_slides = 10

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_fuzzy.assert_not_called()
    mock_socketio.emit.assert_any_call("nav_command", {"action": "NEXT_SLIDE"})


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_highlight_no_cooldown(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """Two rapid fuzzy matches → both emit highlight_text (no cooldown applied)."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "one cycle execution per instruction"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = {
        "slide_index": 2,
        "matched_span": "One cycle execution per instruction",
        "score": 95,
        "emphasis": False,
    }

    voice_app._current_slide_index = 2

    _fill_buffer()
    voice_app._process_speech_buffer()

    _fill_buffer()
    voice_app._process_speech_buffer()

    highlight_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "highlight_text"]
    assert len(highlight_calls) == 2


@patch("app.get_vad_engine")
def test_slide_changed_updates_index(mock_vad):
    """Calling handle_slide_changed updates _current_slide_index."""
    voice_app.handle_slide_changed({"slide_index": 5})
    assert voice_app._current_slide_index == 5


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_interceptor_uses_current_index(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """fuzzy_match_current_slide is called with the correct _current_slide_index."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "one cycle execution per instruction"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None

    voice_app._current_slide_index = 3
    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_fuzzy.assert_called_once_with("one cycle execution per instruction", 3)


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_emphasis_forwarded_in_payload(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy, mock_ctx, mock_vad
):
    """Fuzzy match with emphasis: True → highlight_text payload includes emphasis: True."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "this is very important storage mechanism"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = {
        "slide_index": 1,
        "matched_span": "Physically separate storage for instructions and data",
        "score": 72,
        "emphasis": True,
    }

    voice_app._current_slide_index = 1
    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_socketio.emit.assert_any_call("highlight_text", {
        "slide_index": 1,
        "matched_span": "Physically separate storage for instructions and data",
        "score": 72,
        "emphasis": True,
    })
