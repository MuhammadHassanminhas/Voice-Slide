"""
Tests for qa_assistant.py — Phase 8.

Unit tests for question detection heuristics, notes index construction,
semantic search over slide notes, and pipeline integration.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest
import torch

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import context_search
import qa_assistant
from qa_assistant import is_question, search_notes, build_notes_index
import app as voice_app


# ── Sample Slides for Tests ─────────────────────────────────────────────────

SAMPLE_SLIDES = [
    {
        "id": 1, "type": "title",
        "heading": "Q3 Business Review",
        "subheading": "Presented by Jane Smith",
        "notes": "Welcome everyone to the Q3 business review.",
    },
    {
        "id": 2, "type": "bullets",
        "heading": "Revenue Overview",
        "items": ["Revenue grew 18% year over year"],
        "notes": "Q3 revenue was 680 million dollars, driven by enterprise sales.",
    },
    {
        "id": 3, "type": "bullets",
        "heading": "Market Expansion",
        "items": ["Entered 3 new markets"],
        "notes": "We expanded into Southeast Asia, Latin America, and Eastern Europe.",
    },
    {
        "id": 4, "type": "text",
        "heading": "Team Structure",
        "body": "Our engineering team grew by 40%.",
        "notes": "",   # Empty notes — should be excluded from index
    },
    {
        "id": 5, "type": "quote",
        "heading": "Vision",
        "quote": "The best way to predict the future is to invent it.",
        "attribution": "Alan Kay",
        "notes": "This quote aligns with our innovation-first strategy.",
    },
]


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_qa_state():
    """Reset qa_assistant module state before each test."""
    qa_assistant._note_entries = []
    qa_assistant._note_embeddings = None
    yield


@pytest.fixture
def notes_index():
    """Build context_search and qa_assistant indices from SAMPLE_SLIDES.

    Uses mock embeddings to avoid loading the real model.
    """
    context_search.build_index(SAMPLE_SLIDES)

    # Create deterministic fake embeddings for the 4 notes (slide 4 excluded)
    fake_embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Slide 0: Q3 Business Review (welcome)
        [0.0, 1.0, 0.0, 0.0],  # Slide 1: Revenue Overview (revenue)
        [0.0, 0.0, 1.0, 0.0],  # Slide 2: Market Expansion
        [0.0, 0.0, 0.0, 1.0],  # Slide 4: Vision (innovation)
    ])

    with patch("qa_assistant.encode", return_value=fake_embeddings):
        build_notes_index()

    yield


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
# Question Detection Tests (1–15)
# ══════════════════════════════════════════════════════════════════════════════

def test_question_what_start():
    """WH-word 'what' at start → True."""
    assert is_question("What is the revenue for Q3") is True


def test_question_how_start():
    """WH-word 'how' at start → True."""
    assert is_question("How did we perform this quarter") is True


def test_question_why_start():
    """WH-word 'why' at start → True."""
    assert is_question("Why did expenses increase") is True


def test_question_where_start():
    """WH-word 'where' at start → True."""
    assert is_question("Where are we expanding next year") is True


def test_question_aux_can():
    """Aux-verb 'can' at start → True."""
    assert is_question("Can you explain the budget allocation") is True


def test_question_aux_is():
    """Aux-verb 'is' at start → True."""
    assert is_question("Is the team growing this year") is True


def test_question_aux_does():
    """Aux-verb 'does' at start → True."""
    assert is_question("Does this include international revenue") is True


def test_question_trailing_mark():
    """Trailing '?' → True regardless of word order."""
    assert is_question("the budget is how much?") is True


def test_statement_next_slide():
    """Navigation command without interrogative pattern → False."""
    assert is_question("Next slide please") is False


def test_statement_declarative():
    """Declarative content → False."""
    assert is_question("Revenue grew 18 percent year over year") is False


def test_statement_emphasis():
    """Emphasis phrase without interrogative pattern → False."""
    assert is_question("This is really important remember this") is False


def test_short_filler_what():
    """Single WH-word too short (< 3 words) → False."""
    assert is_question("What") is False


def test_short_filler_how():
    """Two-word WH-phrase too short (< 3 words) → False."""
    assert is_question("How so") is False


def test_empty_string():
    """Empty string → False."""
    assert is_question("") is False


def test_question_who_start():
    """WH-word 'who' at start → True."""
    assert is_question("Who is leading the engineering team") is True


# ── Leading-Punctuation Robustness (STT artifact tolerance) ─────────────────

def test_question_leading_hyphen():
    """Hyphen prefix from faster-whisper → still detected."""
    assert is_question("- Why do we want to be compared") is True


def test_question_leading_bullet():
    """Bullet prefix from faster-whisper → still detected."""
    assert is_question("\u2022 How can we improve our metrics") is True


def test_question_leading_ellipsis():
    """Ellipsis prefix from faster-whisper → still detected."""
    assert is_question("...What is the revenue for Q3") is True


def test_question_leading_smart_quote():
    """Left smart quote prefix → still detected."""
    assert is_question("\u201cWho is leading the engineering team") is True


def test_question_leading_em_dash_and_space():
    """Em-dash + space prefix → still detected."""
    assert is_question("\u2014 Is the team growing this year") is True


def test_statement_leading_hyphen_still_false():
    """Punctuation stripped but declarative content → still False."""
    assert is_question("- Revenue grew 18 percent year over year") is False


# ══════════════════════════════════════════════════════════════════════════════
# Notes Index & Retrieval Tests (16–25)
# ══════════════════════════════════════════════════════════════════════════════

def test_build_notes_index_count(notes_index):
    """build_notes_index() with SAMPLE_SLIDES → exactly 4 entries indexed."""
    assert len(qa_assistant._note_entries) == 4


def test_build_notes_index_excludes_empty(notes_index):
    """Slide 4 (notes: '') is NOT in _note_entries."""
    slide_indices = [e["slide_index"] for e in qa_assistant._note_entries]
    assert 3 not in slide_indices  # slide_index 3 = Slide 4 (0-based)


def test_build_notes_index_entries_have_correct_fields(notes_index):
    """Each entry has slide_index, heading, note_text keys."""
    for entry in qa_assistant._note_entries:
        assert "slide_index" in entry
        assert "heading" in entry
        assert "note_text" in entry


def test_search_notes_returns_list(notes_index):
    """search_notes() returns a list."""
    with patch("qa_assistant.encode") as mock_encode:
        mock_encode.return_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        result = search_notes("revenue")
    assert isinstance(result, list)


def test_search_notes_max_three(notes_index):
    """Even with >3 notes above threshold, result length ≤ 3."""
    # Query that is close to all 4 notes
    with patch("qa_assistant.encode") as mock_encode:
        mock_encode.return_value = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = search_notes("general business overview")
    assert len(result) <= 3


def test_search_notes_results_sorted_by_score(notes_index):
    """Results are sorted by descending score."""
    with patch("qa_assistant.encode") as mock_encode:
        mock_encode.return_value = torch.tensor([[0.1, 0.9, 0.3, 0.0]])
        result = search_notes("what is revenue")
    if len(result) > 1:
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


def test_search_notes_result_fields(notes_index):
    """Each result dict has slide_index, heading, note_text, score."""
    with patch("qa_assistant.encode") as mock_encode:
        mock_encode.return_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        result = search_notes("revenue")
    assert len(result) > 0
    for r in result:
        assert "slide_index" in r
        assert "heading" in r
        assert "note_text" in r
        assert "score" in r


def test_search_notes_empty_index():
    """With no slides loaded → returns []."""
    result = search_notes("anything")
    assert result == []


def test_search_notes_no_match(notes_index):
    """Completely irrelevant query → returns [] (all below threshold)."""
    with patch("qa_assistant.encode") as mock_encode:
        # Orthogonal vector — zero similarity with all notes
        mock_encode.return_value = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        result = search_notes("xyzzy quantum entanglement")
    assert result == []


def test_search_notes_relevant_query(notes_index):
    """Query about revenue → top result is slide_index 1 (Revenue Overview)."""
    with patch("qa_assistant.encode") as mock_encode:
        # Strongly aligned with Revenue Overview embedding (index 1)
        mock_encode.return_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        result = search_notes("What is Q3 revenue")
    assert len(result) > 0
    assert result[0]["slide_index"] == 1
    assert result[0]["heading"] == "Revenue Overview"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration Tests (26–30)
# ══════════════════════════════════════════════════════════════════════════════

@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_question_triggers_qa_update_emit(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy,
    mock_qa, mock_ctx, mock_vad
):
    """Question transcript → qa_update emitted."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "What is the revenue for Q3"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = True
    mock_qa.search_notes.return_value = [
        {"slide_index": 1, "heading": "Revenue Overview",
         "note_text": "Q3 revenue was 680 million dollars.", "score": 0.82},
    ]

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_qa.is_question.assert_called_once_with("What is the revenue for Q3")
    mock_qa.search_notes.assert_called_once_with("What is the revenue for Q3")
    mock_socketio.emit.assert_any_call("qa_update", {
        "question": "What is the revenue for Q3",
        "results": [
            {"slide_index": 1, "heading": "Revenue Overview",
             "note_text": "Q3 revenue was 680 million dollars.", "score": 0.82},
        ],
    })


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_statement_does_not_trigger_qa_update(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy,
    mock_qa, mock_ctx, mock_vad
):
    """Statement transcript → qa_update NOT emitted."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "Next slide please"}
    mock_classify.return_value = {"intent": "NEXT_SLIDE", "confidence": 1.0}
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = False

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_qa.is_question.assert_called_once_with("Next slide please")
    mock_qa.search_notes.assert_not_called()
    qa_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "qa_update"]
    assert len(qa_calls) == 0


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_question_does_not_block_intent_classification(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy,
    mock_qa, mock_ctx, mock_vad
):
    """Question detected → classify_intent() is still called (non-blocking)."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "Can you go to the budget slide"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = True
    mock_qa.search_notes.return_value = []

    _fill_buffer()
    voice_app._process_speech_buffer()

    mock_classify.assert_called_once_with("Can you go to the budget slide")


@patch("app.get_vad_engine")
@patch("app.context_search")
@patch("app.qa_assistant")
@patch("app.fuzzy_match_current_slide")
@patch("app.classify_intent")
@patch("app.transcribe_chunk")
@patch("app.socketio")
def test_qa_update_payload_structure(
    mock_socketio, mock_transcribe, mock_classify, mock_fuzzy,
    mock_qa, mock_ctx, mock_vad
):
    """Verify qa_update payload has question (str) and results (list of dicts)."""
    mock_vad.return_value.reset = MagicMock()
    mock_transcribe.return_value = {"text": "How did we perform this quarter"}
    mock_classify.return_value = {"intent": "NONE", "confidence": 0.0}
    mock_fuzzy.return_value = None
    mock_ctx.search.return_value = None
    mock_ctx._num_slides = 10
    mock_qa.is_question.return_value = True
    mock_qa.search_notes.return_value = [
        {"slide_index": 0, "heading": "Q3 Business Review",
         "note_text": "Welcome everyone to the Q3 business review.", "score": 0.55},
    ]

    _fill_buffer()
    voice_app._process_speech_buffer()

    qa_calls = [c for c in mock_socketio.emit.call_args_list if c[0][0] == "qa_update"]
    assert len(qa_calls) == 1
    payload = qa_calls[0][0][1]
    assert isinstance(payload["question"], str)
    assert isinstance(payload["results"], list)
    assert all(isinstance(r, dict) for r in payload["results"])


def test_presenter_route_returns_200():
    """GET /presenter → status 200."""
    client = voice_app.app.test_client()
    response = client.get("/presenter")
    assert response.status_code == 200
