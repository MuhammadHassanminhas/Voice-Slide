"""
Tests for the Universal Semantic Fallback — Phase 5 Refactor.

These integration tests verify the fallback path in _process_speech_buffer():
when the intent classifier returns NONE, the raw transcribed text is passed
to context_search.search() and, if a match is found, GOTO_CONTENT is emitted.
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
def reset_cooldown():
    """Reset the nav command cooldown and speech buffer before each test."""
    voice_app._last_nav_command_time = 0.0
    voice_app._speech_buffer.clear()
    voice_app._silence_counter = 0
    voice_app._is_speaking = False
    yield


def _fill_buffer():
    """Put dummy audio bytes into the speech buffer so _process_speech_buffer runs."""
    voice_app._speech_buffer.extend(b"\x00" * 100)


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_fallback_emits_goto_content(mock_socketio, mock_transcribe, mock_classify, mock_ctx, mock_vad):
    """Classifier returns NONE + context_search finds a match → emit GOTO_CONTENT."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "the CISC architecture"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_ctx.search.return_value = {
        "slide_index": 4,
        "score": 0.42,
        "matched_text": "CISC architecture details...",
    }

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_socketio.emit.assert_any_call("nav_command", {
        "action": "GOTO_CONTENT",
        "slide_number": 5,
    })


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_fallback_no_match_no_emit(mock_socketio, mock_transcribe, mock_classify, mock_ctx, mock_vad):
    """Classifier returns NONE + context_search finds nothing → no nav_command emitted."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "that is a great point"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_ctx.search.return_value = None

    _fill_buffer()
    voice_app._process_speech_buffer()

    nav_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "nav_command"]
    assert len(nav_calls) == 0


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_recognized_command_skips_fallback(mock_socketio, mock_transcribe, mock_classify, mock_ctx, mock_vad):
    """Classifier returns NEXT_SLIDE → emit immediately, never call context_search."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "next slide"}
    mock_classify.return_value = {"intent": "NEXT_SLIDE", "confidence": 1.0}
    mock_ctx._num_slides = 10

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_ctx.search.assert_not_called()
    mock_socketio.emit.assert_any_call("nav_command", {"action": "NEXT_SLIDE"})


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_goto_slide_without_number_uses_context(mock_socketio, mock_transcribe, mock_classify, mock_ctx, mock_vad):
    """GOTO_SLIDE without a slide_number falls through to context_search."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "go to the budget slide"}
    mock_classify.return_value = {"intent": "GOTO_SLIDE", "confidence": 0.75}
    mock_ctx.search.return_value = {
        "slide_index": 7,
        "score": 0.38,
        "matched_text": "Budget overview...",
    }

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_ctx.search.assert_called_once()
    mock_socketio.emit.assert_any_call("nav_command", {
        "action": "GOTO_CONTENT",
        "slide_number": 8,
    })


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_fallback_respects_cooldown(mock_socketio, mock_transcribe, mock_classify, mock_ctx, mock_vad):
    """Fallback match should be suppressed if within the cooldown window."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "the CISC architecture"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_ctx.search.return_value = {
        "slide_index": 4,
        "score": 0.42,
        "matched_text": "CISC architecture details...",
    }

    # Set last command time to "just now" to trigger cooldown
    voice_app._last_nav_command_time = time.time()

    _fill_buffer()
    voice_app._process_speech_buffer()

    nav_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "nav_command"]
    assert len(nav_calls) == 0
