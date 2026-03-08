"""
Tests for keyword_highlighter.py — Phase 6.

Unit tests for fuzzy matching logic, slide text span extraction,
emphasis detection, and edge cases.
"""

import sys
import os

import pytest

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import context_search
from keyword_highlighter import (
    fuzzy_match_current_slide,
    _get_slide_text_spans,
    _detect_emphasis,
)


# ── Sample Slides for Tests ─────────────────────────────────────────────────

SAMPLE_SLIDES = [
    {
        "id": 1, "type": "title",
        "heading": "Q3 Business Review",
        "subheading": "Presented by Jane Smith",
        "notes": "Welcome everyone.",
    },
    {
        "id": 2, "type": "bullets",
        "heading": "Harvard Architecture",
        "items": [
            "Physically separate storage for instructions and data",
            "CPU can access instructions and read/write data simultaneously",
        ],
        "notes": "",
    },
    {
        "id": 3, "type": "bullets",
        "heading": "Features of RISC Processors",
        "items": [
            "One cycle execution per instruction",
            "Pipelining for better efficiency",
            "Large register set",
        ],
        "notes": "",
    },
    {
        "id": 4, "type": "image",
        "heading": "Revenue Chart",
        "caption": "Revenue grew 18% year over year",
        "image_url": "/static/images/chart.png",
        "notes": "Q3 revenue was 680 million dollars.",
    },
    {
        "id": 5, "type": "two-column",
        "heading": "Team Structure",
        "left": {"title": "Engineering", "items": ["Backend", "Frontend", "DevOps"]},
        "right": {"title": "Product", "items": ["Design", "Research", "Analytics"]},
        "notes": "",
    },
    {
        "id": 6, "type": "quote",
        "heading": "Inspiration",
        "quote": "The best way to predict the future is to invent it.",
        "attribution": "Alan Kay",
        "notes": "",
    },
]


@pytest.fixture(autouse=True)
def index_sample_slides():
    """Build the context search index with sample slides before each test."""
    context_search.build_index(SAMPLE_SLIDES)
    yield
    context_search.build_index([])


# ── Fuzzy Matching ──────────────────────────────────────────────────────────

def test_exact_bullet_match():
    """Transcript exactly matching a bullet returns a highlight."""
    result = fuzzy_match_current_slide(
        "Physically separate storage for instructions and data", 1
    )
    assert result is not None
    assert result["score"] >= 65
    assert result["slide_index"] == 1


def test_partial_bullet_match():
    """Transcript matching the first half of a bullet still matches."""
    result = fuzzy_match_current_slide(
        "one cycle execution per instruction", 2
    )
    assert result is not None
    assert result["score"] >= 65


def test_heading_match():
    """Transcript matching slide heading returns highlight."""
    result = fuzzy_match_current_slide(
        "the features of RISC processors", 2
    )
    assert result is not None
    assert result["matched_span"] == "Features of RISC Processors"


def test_caption_match():
    """Transcript matching image caption returns highlight."""
    result = fuzzy_match_current_slide(
        "revenue grew 18 percent year over year", 3
    )
    assert result is not None
    assert result["slide_index"] == 3


def test_quote_match():
    """Transcript matching quote text returns highlight."""
    result = fuzzy_match_current_slide(
        "the best way to predict the future is to invent it", 5
    )
    assert result is not None
    assert result["slide_index"] == 5


def test_two_column_match():
    """Transcript matching a column item returns highlight."""
    result = fuzzy_match_current_slide(
        "backend frontend and DevOps teams", 4
    )
    assert result is not None
    assert result["slide_index"] == 4


def test_no_match_unrelated():
    """Transcript unrelated to the current slide returns None."""
    result = fuzzy_match_current_slide(
        "quantum physics and dark matter theory", 1
    )
    assert result is None


def test_short_transcript_guard():
    """Transcript with fewer than 3 words returns None."""
    result = fuzzy_match_current_slide("hello world", 1)
    assert result is None


def test_empty_slide_no_crash():
    """Slide with no text spans returns None."""
    context_search._slides = [{"heading": "", "items": []}]
    result = fuzzy_match_current_slide("some words about nothing here", 0)
    assert result is None


def test_invalid_slide_index():
    """Out-of-range slide_index returns None."""
    result = fuzzy_match_current_slide(
        "separate storage for instructions and data", 99
    )
    assert result is None


def test_no_slides_loaded():
    """No slides loaded returns None."""
    context_search._slides = []
    result = fuzzy_match_current_slide(
        "separate storage for instructions and data", 0
    )
    assert result is None


def test_emphasis_detected():
    """Transcript with 'this is very important' sets emphasis True."""
    result = fuzzy_match_current_slide(
        "physically separate storage this is very important", 1
    )
    assert result is not None
    assert result["emphasis"] is True


def test_emphasis_not_detected():
    """Normal transcript without emphasis phrase sets emphasis False."""
    result = fuzzy_match_current_slide(
        "physically separate storage for instructions and data", 1
    )
    assert result is not None
    assert result["emphasis"] is False


def test_emphasis_key_point():
    """Transcript with 'key point' sets emphasis True."""
    result = fuzzy_match_current_slide(
        "one cycle execution key point to remember", 2
    )
    assert result is not None
    assert result["emphasis"] is True


def test_best_span_returned():
    """When multiple spans match, the highest-scoring one is returned."""
    result = fuzzy_match_current_slide(
        "pipelining for better efficiency", 2
    )
    assert result is not None
    assert result["matched_span"] == "Pipelining for better efficiency"


def test_case_insensitive():
    """Mixed-case transcript matches regardless of casing."""
    result = fuzzy_match_current_slide(
        "PHYSICALLY SEPARATE STORAGE FOR INSTRUCTIONS", 1
    )
    assert result is not None
    assert result["score"] >= 65


# ── Emphasis Regex ──────────────────────────────────────────────────────────

def test_emphasis_remember_this():
    assert _detect_emphasis("remember this detail about storage") is True


def test_emphasis_pay_attention():
    assert _detect_emphasis("pay attention to this architecture") is True


def test_emphasis_really_important():
    assert _detect_emphasis("this is really important for the exam") is True


def test_emphasis_no_trigger():
    assert _detect_emphasis("the pipeline has five stages") is False
