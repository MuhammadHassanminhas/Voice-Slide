import sys
import os
import pytest

# Add backend to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from intent_classifier import classify_intent, _initialize_classifier


# ── Fast-Path Exact Matches ──────────────────────────────────────────────────

def test_fast_path_next_slide():
    """Exact canonical phrase should return confidence 1.0 via fast-path."""
    res = classify_intent("next slide")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] == 1.0

def test_fast_path_go_back():
    res = classify_intent("go back")
    assert res['intent'] == "PREV_SLIDE"
    assert res['confidence'] == 1.0

def test_fast_path_case_insensitive():
    res = classify_intent("Next Slide")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] == 1.0


# ── Embedding-Based Semantic Matches ─────────────────────────────────────────

def test_intent_classifier_exact_matches():
    _initialize_classifier()

    res = classify_intent("next slide")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] > 0.90

    res = classify_intent("go back")
    assert res['intent'] == "PREV_SLIDE"
    assert res['confidence'] > 0.90

def test_intent_classifier_semantic_matches():
    _initialize_classifier()

    res = classify_intent("Let's move on to the next one")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] >= 0.55

    res = classify_intent("Could you show the previous slide again")
    assert res['intent'] == "PREV_SLIDE"
    assert res['confidence'] >= 0.55


# ── GOTO_SLIDE with Number Extraction ────────────────────────────────────────

def test_goto_slide_with_digit():
    res = classify_intent("go to slide 3")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 3

def test_goto_slide_with_word_number():
    res = classify_intent("jump to slide five")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 5

def test_goto_slide_semantic():
    _initialize_classifier()
    res = classify_intent("can we skip to slide number 7")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 7


# ── NEXT_POINT / PREV_POINT ─────────────────────────────────────────────────

def test_next_point():
    res = classify_intent("next point")
    assert res['intent'] == "NEXT_POINT"

def test_prev_point():
    res = classify_intent("previous bullet")
    assert res['intent'] == "PREV_POINT"


# ── Non-Command Speech → NONE ───────────────────────────────────────────────

def test_intent_classifier_no_match():
    _initialize_classifier()

    res = classify_intent("This is a really great point about quarter three revenue")
    assert res['intent'] == "NONE"

    res = classify_intent("")
    assert res['intent'] == "NONE"

    res = classify_intent("   ")
    assert res['intent'] == "NONE"


# ── Tier 0a: Length Penalty ─────────────────────────────────────────────────

def test_length_penalty_long_sentence():
    """15-word sentence containing 'next slide' should be rejected immediately."""
    res = classify_intent(
        "I think we should probably move on to discuss the next slide in our presentation"
    )
    assert res['intent'] == "NONE"
    assert res['confidence'] == 0.0

def test_length_penalty_boundary_11_words():
    """12 words (above the >10 boundary) should be rejected."""
    res = classify_intent("I really think we should go ahead and go to next slide")
    assert res['intent'] == "NONE"
    assert res['confidence'] == 0.0

def test_length_penalty_boundary_10_words_passes():
    """Exactly 10 words should NOT be rejected (check is > 10, not >= 10)."""
    res = classify_intent("I think we should now go to the next slide")
    assert res['intent'] != "NONE"


# ── Tier 0b: End-of-Sentence Bias ──────────────────────────────────────────

def test_endswith_next_slide():
    res = classify_intent("please go to the next slide")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] == 0.95

def test_endswith_go_back():
    res = classify_intent("can we go back")
    assert res['intent'] == "PREV_SLIDE"
    assert res['confidence'] == 0.95

def test_endswith_continue():
    res = classify_intent("alright let's continue")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] == 0.95

def test_endswith_move_forward():
    res = classify_intent("okay please move forward")
    assert res['intent'] == "NEXT_SLIDE"
    assert res['confidence'] == 0.95


# ── False Positive Rejection ───────────────────────────────────────────────

def test_false_positive_mid_sentence():
    """'next slide' at the START followed by more words should NOT trigger."""
    _initialize_classifier()
    res = classify_intent("the next slide shows our revenue")
    assert res['intent'] == "NONE"

def test_false_positive_previous_slide_start():
    """'previous slide' at the START followed by more words should NOT trigger."""
    _initialize_classifier()
    res = classify_intent("previous slide had a great chart")
    assert res['intent'] == "NONE"

def test_false_positive_go_back_mid_sentence():
    """'go back' in the MIDDLE of a sentence should NOT trigger."""
    _initialize_classifier()
    res = classify_intent("if we go back to the data")
    assert res['intent'] == "NONE"


# ── Ordinal Number Extraction ──────────────────────────────────────────────

def test_goto_slide_ordinal_first():
    _initialize_classifier()
    res = classify_intent("go to the first slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 1

def test_goto_slide_ordinal_third():
    """The exact failure case reported by the user."""
    _initialize_classifier()
    res = classify_intent("move to the third slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 3

def test_goto_slide_ordinal_tenth():
    _initialize_classifier()
    res = classify_intent("jump to the tenth slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 10

def test_goto_slide_ordinal_twentieth():
    _initialize_classifier()
    res = classify_intent("show the twentieth slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 20

def test_goto_slide_ordinal_fifth():
    _initialize_classifier()
    res = classify_intent("can we skip to the fifth slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 5


# ── Word Boundary Safety (Substring Bug Fix) ──────────────────────────────

def test_extract_eighteen_not_eight():
    """'eighteen' must NOT be matched as 'eight' by the cardinal scan."""
    _initialize_classifier()
    res = classify_intent("go to slide eighteen")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 18

def test_extract_fourteenth_not_four():
    """'fourteenth' must NOT be matched as 'four' by the cardinal scan."""
    _initialize_classifier()
    res = classify_intent("go to the fourteenth slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 14


# ── Digit-Ordinal Extraction (e.g. "4th", "1st") ─────────────────────────

def test_goto_slide_digit_ordinal_4th():
    """The exact bug: '4th' was not matched by the pure-digit regex."""
    _initialize_classifier()
    res = classify_intent("move to the 4th slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 4

def test_goto_slide_digit_ordinal_1st():
    _initialize_classifier()
    res = classify_intent("go to the 1st slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 1

def test_goto_slide_digit_ordinal_2nd():
    _initialize_classifier()
    res = classify_intent("jump to the 2nd slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 2

def test_goto_slide_digit_ordinal_3rd():
    _initialize_classifier()
    res = classify_intent("switch to the 3rd slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 3

def test_goto_slide_digit_ordinal_10th():
    _initialize_classifier()
    res = classify_intent("skip to the 10th slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 10

def test_goto_slide_pure_digit_still_works():
    """Regression gate: pure digit without suffix must still work."""
    res = classify_intent("go to slide 5")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 5


# ── Natural Content Queries → NONE (Universal Fallback) ──────────────────

def test_natural_query_returns_none_cisc():
    """Conversational content query must return NONE (handled by app.py fallback)."""
    _initialize_classifier()
    res = classify_intent("let's look at the CISC architecture")
    assert res['intent'] == "NONE"

def test_natural_query_returns_none_harvard():
    _initialize_classifier()
    res = classify_intent("what about the Harvard stuff")
    assert res['intent'] == "NONE"

def test_natural_query_returns_none_show_me():
    """Former prefix phrase must also return NONE — no longer special-cased."""
    _initialize_classifier()
    res = classify_intent("show me the Harvard architecture")
    assert res['intent'] == "NONE"

def test_natural_query_returns_none_revisit():
    _initialize_classifier()
    res = classify_intent("let's revisit the RISC features")
    assert res['intent'] == "NONE"


# ── GOTO_SLIDE Regression Gates ──────────────────────────────────────────

def test_goto_slide_basic_with_number():
    """GOTO_SLIDE with an explicit number must always be recognized."""
    res = classify_intent("go to slide 3")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 3

def test_goto_slide_ordinal_basic():
    """Digit-ordinal GOTO_SLIDE must always be recognized."""
    _initialize_classifier()
    res = classify_intent("jump to the 4th slide")
    assert res['intent'] == "GOTO_SLIDE"
    assert res['slide_number'] == 4
