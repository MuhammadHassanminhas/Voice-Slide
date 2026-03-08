import sys
import os
import pytest

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import context_search
from context_search import _extract_title, _extract_body, _clean_query


# ── Sample Slides ────────────────────────────────────────────────────────────

SAMPLE_SLIDES = [
    {
        "id": 1, "type": "title",
        "heading": "Q3 Business Review",
        "subheading": "Presented by Jane Smith",
        "notes": "Welcome everyone to the quarterly review.",
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


# ── Query Cleaning ──────────────────────────────────────────────────────────

def test_clean_query_strips_nav_verbs():
    assert _clean_query("move to the RISC slide") == "RISC"

def test_clean_query_strips_show_me():
    assert _clean_query("show me the Harvard architecture") == "Harvard architecture"

def test_clean_query_strips_compound_phrase():
    assert _clean_query("can you take me to the part about pipelining") == "pipelining"

def test_clean_query_preserves_bare_topic():
    assert _clean_query("CISC architecture") == "CISC architecture"

def test_clean_query_empty_result():
    """All words are noise — returns empty string."""
    assert _clean_query("go to the slide") == ""

def test_clean_query_strips_where_was():
    assert _clean_query("where was the Von Neumann thing") == "Von Neumann"


# ── Title Extraction ────────────────────────────────────────────────────────

def test_extract_title_heading_and_subheading():
    result = _extract_title({"heading": "Q3 Review", "subheading": "By Jane"})
    assert result == "Q3 Review By Jane"

def test_extract_title_heading_only():
    result = _extract_title({"heading": "Harvard Architecture"})
    assert result == "Harvard Architecture"

def test_extract_title_empty():
    """Slide with no meaningful heading returns empty string."""
    result = _extract_title({"heading": "", "items": ["bullet"]})
    assert result == ""


# ── Body Extraction ─────────────────────────────────────────────────────────

def test_extract_body_bullets():
    result = _extract_body({"heading": "Title", "items": ["a", "b"]})
    assert result == "a b"
    assert "Title" not in result

def test_extract_body_two_column():
    result = _extract_body(SAMPLE_SLIDES[4])
    assert "Engineering" in result
    assert "Product" in result
    assert "Backend" in result
    assert "Analytics" in result

def test_extract_body_empty():
    result = _extract_body({"heading": "Title Only"})
    assert result == ""


# ── build_index ──────────────────────────────────────────────────────────────

def test_build_index_populates_embeddings():
    """build_index with valid slides creates dual embeddings."""
    context_search.build_index(SAMPLE_SLIDES[:3])
    assert context_search._title_embeddings is not None
    assert context_search._body_embeddings is not None
    assert context_search._num_slides == 3


def test_build_index_empty_slides():
    """build_index with empty list results in no embeddings."""
    context_search.build_index([])
    assert context_search._title_embeddings is None
    assert context_search._body_embeddings is None
    assert context_search._num_slides == 0


def test_build_index_dual_tensors():
    """Both title and body tensors have the correct number of rows."""
    context_search.build_index(SAMPLE_SLIDES)
    assert context_search._title_embeddings.shape[0] == 6
    assert context_search._body_embeddings.shape[0] == 6


# ── search ───────────────────────────────────────────────────────────────────

def test_search_exact_heading_match():
    """Query matching a slide heading returns that slide."""
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("Harvard Architecture")
    assert result is not None
    assert result["slide_index"] == 1  # SAMPLE_SLIDES[1]
    assert result["score"] >= 0.30


def test_search_partial_topic_match():
    """Query with partial topic terms finds the right slide."""
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("RISC processors pipelining")
    assert result is not None
    assert result["slide_index"] == 2  # SAMPLE_SLIDES[2]


def test_search_no_match_below_threshold():
    """Query unrelated to any slide returns None."""
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("quantum physics dark matter")
    assert result is None


def test_search_empty_index():
    """search on an empty index returns None."""
    context_search.build_index([])
    result = context_search.search("Harvard Architecture")
    assert result is None


def test_reindex_updates_embeddings():
    """Calling build_index again replaces the old index."""
    context_search.build_index(SAMPLE_SLIDES[:2])
    # Slide 2 (RISC) is NOT in the index
    result = context_search.search("RISC processors pipelining")
    # Might match or not, but now rebuild with all slides
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("RISC processors pipelining")
    assert result is not None
    assert result["slide_index"] == 2


# ── Dual-Index Edge Cases ───────────────────────────────────────────────────

def test_body_match_works_for_detail_query():
    """Detailed body-level query still finds the right slide via body channel."""
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("separate storage for instructions and data")
    assert result is not None
    assert result["slide_index"] == 1  # Harvard Architecture body

def test_missing_title_no_crash():
    """Slide with empty heading should not crash — body channel picks it up."""
    context_search.build_index([{"heading": "", "items": ["bullet text about testing"]}])
    result = context_search.search("bullet text about testing")
    assert result is not None
    assert result["slide_index"] == 0

def test_missing_body_no_crash():
    """Slide with no body content should not crash — title channel picks it up."""
    context_search.build_index([{"heading": "Solo Title"}])
    result = context_search.search("Solo Title")
    assert result is not None
    assert result["slide_index"] == 0

def test_noisy_query_finds_correct_slide():
    """End-to-end: noisy voice query cleaned + dual-index → correct slide."""
    context_search.build_index(SAMPLE_SLIDES)
    result = context_search.search("move to the RISC slide")
    assert result is not None
    assert result["slide_index"] == 2  # RISC Processors

def test_title_match_beats_body_dilution():
    """A concise topic query should score higher via the title channel."""
    slides = [
        {
            "heading": "Pipelining",
            "items": [
                "This is a long description about how processors execute",
                "multiple stages of instructions in an overlapping fashion",
                "to improve throughput and overall system performance",
                "which is a fundamental technique in modern CPU design",
            ],
        },
    ]
    context_search.build_index(slides)
    result = context_search.search("pipelining")
    assert result is not None
    assert result["score"] >= 0.30
