# Phase 4.2 — Implementation Plan: Intent Classifier Heuristics & Ordinal Extraction

> **Status**: Plan for review — no code will be written until approved.
>
> **Objective**: Harden the intent classifier against false positives from full-utterance
> transcripts (a side-effect of the VAD pipeline shipping complete sentences) and add
> ordinal-word support so `"move to the third slide"` resolves to slide 3.

---

## 0. Current Code Snapshot (What We're Changing)

### `backend/intent_classifier.py` — 158 lines

| Symbol | Lines | Current Behavior |
|---|---|---|
| `CANONICAL_INTENTS` | 20–47 | 7 intents, 30 canonical phrases |
| `FAST_PATH_MAP` | 55–58 | Flat `dict[str, str]` built from `CANONICAL_INTENTS`. Keyed by phrase → intent. |
| `_fast_path_match(text)` | 61–67 | **Exact-only** match: `FAST_PATH_MAP.get(normalized)`. Returns `{"intent", "confidence": 1.0}` or `None`. |
| `WORD_TO_NUM` | 72–77 | Cardinals only: `"one"→1` … `"twenty"→20`. **No ordinals.** |
| `_extract_slide_number(text)` | 80–89 | First tries `\b(\d+)\b` regex, then scans `WORD_TO_NUM` keys with bare `in` check. |
| `CONFIDENCE_THRESHOLD` | 50 | `0.55` — unchanged. |
| `classify_intent(text)` | 118–158 | Flow: empty guard → Tier 1 fast-path → Tier 2 embedding → NONE. |

### `tests/test_intent_classifier.py` — 97 lines, 11 tests

| Group | Count | Coverage |
|---|---|---|
| Fast-path exact match | 3 | `"next slide"`, `"go back"`, `"Next Slide"` (case) |
| Embedding semantic | 2 | `"Let's move on to the next one"`, `"Could you show the previous slide again"` |
| GOTO_SLIDE extraction | 3 | Digit `"3"`, cardinal word `"five"`, semantic + digit `"7"` |
| NEXT/PREV_POINT | 2 | `"next point"`, `"previous bullet"` |
| No-match / NONE | 1 | Long sentence, empty string, whitespace |

**Gaps**: No ordinal tests. No heuristic false-positive tests. No end-of-sentence bias tests.

---

## 1. Phase 4.1 Heuristics — Detailed Integration Plan

The heuristics form a new **Tier 0** layer that runs *before* the existing fast-path and
embedding lookup. The goal is to pre-filter text that is structurally unlikely to be a
voice command, preventing false positives that became possible now that the VAD pipeline
delivers complete sentences instead of 250ms fragments.

### 1.1 Tier 0a — 10-Word Length Penalty

**Rule**: If the transcribed text contains more than 10 whitespace-delimited words,
return `{"intent": "NONE", "confidence": 0.0}` immediately.

**Rationale**: Real voice commands are short imperatives (2–6 words). The longest
reasonable command in our canonical set is 7 words (`"Could you show the previous slide
again"`). Anything over 10 words is overwhelmingly natural speech, not a navigation
command.

**Insertion point**: `classify_intent()`, line 128 — immediately after the existing
empty-string guard (line 126–127) and before the Tier 1 fast-path call (line 130).

```
classify_intent(text)                    # line 118
│
├── Guard: empty / whitespace → NONE     # lines 126-127  (EXISTING — unchanged)
│
├── NEW ▸ Tier 0a: Length Penalty        # INSERT HERE (new lines ~128-131)
│   └── if len(text.split()) > 10 → return NONE
│
├── Tier 1: _fast_path_match(text)       # line 130  (EXISTING — will shift down ~4 lines)
├── Tier 2: embedding cosine similarity  # line 137+ (EXISTING — unchanged)
└── Return NONE                          # line 158  (EXISTING — unchanged)
```

**Planned code** (4 lines):

```python
# ── Tier 0a: Length Penalty ──────────────────────────────────────────
words = text.split()
if len(words) > 10:
    logger.debug("Length penalty: '%s' has %d words → NONE", text, len(words))
    return {"intent": "NONE", "confidence": 0.0}
```

**Impact on existing tests**: Zero. The longest test input is `"This is a really great
point about quarter three revenue"` at exactly 10 words — the `> 10` check will NOT
filter it. It will continue falling through to NONE via the embedding path as before.

---

### 1.2 Tier 0b — End-of-Sentence Bias (Fast-Path Modification)

**Problem**: With the VAD pipeline, the transcriber now delivers full utterances. A user
saying *"please go to the next slide"* produces a 7-word sentence. The current fast-path
only does exact-match, so it won't catch this — it falls through to the embedding path,
which might match but at lower confidence. Meanwhile, *"the next slide shows our revenue"*
could also reach the embedding path and falsely trigger.

**Rule**: Extend `_fast_path_match()` to check whether the text **ends with** any
canonical phrase (not just equals it). Command phrases at the end of a sentence are
almost always intentional commands. Command phrases at the start or middle, followed by
more words, are almost always natural speech.

**Modified `_fast_path_match(text)` logic**:

```python
def _fast_path_match(text: str) -> dict | None:
    """
    O(1) exact-match, then O(n) end-of-sentence scan.
    Returns None if no match.
    """
    normalized = text.strip().lower()

    # ── 1. Exact whole-string match (existing behavior, confidence 1.0) ──
    intent = FAST_PATH_MAP.get(normalized)
    if intent:
        return {"intent": intent, "confidence": 1.0}

    # ── 2. End-of-sentence match (NEW, confidence 0.95) ──
    #    Check if normalized text ENDS WITH any canonical phrase.
    #    Require a word boundary (space or start-of-string) before the phrase
    #    to prevent substring matches like "context" matching "text".
    for phrase, intent in FAST_PATH_MAP.items():
        if normalized.endswith(phrase):
            prefix = normalized[: -len(phrase)]
            if prefix == "" or prefix.endswith(" "):
                return {"intent": intent, "confidence": 0.95}

    return None
```

**Key design decisions**:

| Decision | Rationale |
|---|---|
| Confidence `0.95` for end-of-sentence matches | Distinguishes from exact-match (`1.0`) in logs. Still well above the `0.55` embedding threshold. |
| Word-boundary check via `prefix.endswith(" ")` | Prevents `"context"` from matching `"text"`, `"forecast"` from matching `"cast"`, etc. |
| Iterate `FAST_PATH_MAP.items()` (30 entries) | O(n) where n=30 is effectively O(1) for practical purposes. No performance concern. |
| No changes to the embedding path (Tier 2) | The length penalty (Tier 0a) already filters long sentences. Short sentences (≤10 words) that reach the embedding path should still trigger if semantically close — the embedding model naturally scores lower for sentences where the command sense is diluted. |

**Handling GOTO_SLIDE with end-of-sentence match**: The GOTO_SLIDE canonical phrases are
prefix-style (`"go to slide"`, `"jump to slide"`, etc.) — they expect a number *after*
them. The `endswith` check will NOT match these because the number word/digit follows the
phrase. This is correct behavior: `"go to slide 3"` matches via exact fast-path after
`_extract_slide_number` processes it. For sentences like `"please go to slide five"`, the
fast-path won't match (it's not an exact match and doesn't end with a bare canonical
phrase). It will fall through to the embedding path, which will match GOTO_SLIDE, and then
`_extract_slide_number` will extract `5`. **No special handling needed.**

**Wait — re-examining this**: Actually, `"go to slide 3"` does NOT exact-match the
fast-path because `FAST_PATH_MAP` contains `"go to slide"` (without the number). Let me
re-check the current flow:

- Input: `"go to slide 3"`
- `_fast_path_match("go to slide 3")` → `FAST_PATH_MAP.get("go to slide 3")` → `None`
  (because the key is `"go to slide"`, not `"go to slide 3"`)
- Falls through to embedding path → matches GOTO_SLIDE → `_extract_slide_number` extracts `3`

So GOTO_SLIDE already relies on the embedding path for digit/word-suffixed inputs. The
end-of-sentence bias won't interfere because `"go to slide 3"` doesn't end with any
canonical phrase. ✅

For `"please jump to slide five"`:
- Fast-path exact: miss
- Fast-path endswith: `"jump to slide"` — no, it ends with `"slide five"` not `"slide"`.
  Actually checking: does it end with `"jump to slide"`? `"please jump to slide five"` ends
  with `"slide five"`, not `"jump to slide"`. So no endswith match. ✅
- Falls to embedding → GOTO_SLIDE match → extract `5` from `"five"`. Correct.

**Impact on existing tests**: All 11 tests continue to pass. The end-of-sentence scan
only adds *new* matches for inputs that previously fell through to the embedding path or
returned NONE. No existing exact-match behavior is altered.

---

## 2. Ordinal Number Extraction — Detailed Strategy

### 2.1 The Problem

The current `_extract_slide_number()` handles:
- ✅ Digits: `"go to slide 3"` → `3`
- ✅ Cardinal words: `"jump to slide five"` → `5`
- ❌ Ordinal words: `"move to the third slide"` → `None` (FAILS)

Users naturally say ordinals when referring to slide positions: *"go to the first slide"*,
*"show me the third one"*, *"jump to the fifth slide"*.

### 2.2 The Mapping

Add an `ORDINAL_TO_NUM` dictionary alongside the existing `WORD_TO_NUM`:

```python
ORDINAL_TO_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
}
```

**Coverage**: 1st through 20th. This matches the range of the existing `WORD_TO_NUM`
cardinal map. Presentations with more than 20 slides can still be navigated via digit
input (`"go to slide 25"`).

**Why a separate dict instead of merging into `WORD_TO_NUM`?**: Semantic clarity. Cardinals
and ordinals are linguistically distinct word classes. Keeping them separate makes the code
self-documenting and simplifies debugging. Both maps are scanned in `_extract_slide_number`.

### 2.3 Updated `_extract_slide_number()` Logic

```python
def _extract_slide_number(text: str) -> int | None:
    """
    Extract a slide number from text.
    Priority: digit literal → cardinal word → ordinal word.
    """
    # 1. Digit literal (highest priority, unambiguous)
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))

    lower = text.lower()

    # 2. Cardinal word ("five" → 5)
    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            return num

    # 3. Ordinal word ("third" → 3)  ← NEW
    for word, num in ORDINAL_TO_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            return num

    return None
```

**Critical fix — word boundary enforcement**: The current code uses bare `if word in lower`
for cardinal scanning (line 87). This is a latent substring bug:

- `"eighteen"` contains `"eight"` → `"go to slide eighteen"` could return `8` instead of `18`
- `"fourteen"` contains `"four"` → same problem

The fix: replace `if word in lower` with `if re.search(rf"\b{word}\b", lower)` using
word-boundary anchors. This applies to **both** the existing cardinal scan and the new
ordinal scan.

**Priority order rationale**:

| Priority | Source | Example | Why |
|---|---|---|---|
| 1 (highest) | Digit literal | `"slide 3"` | Unambiguous, zero interpretation needed |
| 2 | Cardinal word | `"slide five"` | Direct number reference |
| 3 | Ordinal word | `"the third slide"` | Positional reference, equally valid |

If text contains both a digit and a word (unlikely but possible), the digit wins. This
matches user expectation: `"go to slide 3, the third one"` → `3`.

### 2.4 Interaction with Tier 0b (End-of-Sentence Bias)

Ordinal commands often have the number *before* the word "slide":
- `"go to the third slide"` — ends with `"slide"`, but `"slide"` alone is not a canonical phrase.
- `"move to the fifth one"` — doesn't end with any canonical phrase.

Both of these will fall through to the embedding path (Tier 2), where they'll match
GOTO_SLIDE semantically, and then `_extract_slide_number` will extract the ordinal. The
heuristics layer does not interfere. ✅

### 2.5 Edge Cases Considered

| Input | Expected | Handling |
|---|---|---|
| `"go to the first slide"` | `slide_number: 1` | Ordinal map: `"first"→1` |
| `"move to the twentieth slide"` | `slide_number: 20` | Ordinal map: `"twentieth"→20` |
| `"jump to slide 21"` | `slide_number: 21` | Digit regex: `21` |
| `"skip to slide twenty"` | `slide_number: 20` | Cardinal map: `"twenty"→20` |
| `"go to the second to last slide"` | `slide_number: 2` | Ordinal map: `"second"→2`. This is an incorrect interpretation, but handling relative phrases like "second to last" is out of scope. Noted as a known limitation. |
| `"the third"` (no "slide" word) | `slide_number: 3` | Still extracted. The intent classifier decides if it's GOTO_SLIDE; extraction just finds the number. |
| `"fourteenth"` vs `"four"` | `slide_number: 14` | Word-boundary regex prevents `"four"` from matching inside `"fourteenth"`. Cardinal scan runs first, but `\bfour\b` won't match `"fourteenth"`. Ordinal scan then matches `\bfourteenth\b` → `14`. ✅ |
| `"eighteen"` vs `"eight"` | `slide_number: 18` | Same word-boundary fix. `\beight\b` won't match `"eighteen"`. `\beighteen\b` matches → `18`. ✅ |

---

## 3. Complete Updated `classify_intent()` Flow

After all changes, the full classification flow becomes:

```
classify_intent(text)
│
├── Guard: empty / whitespace → NONE (confidence 0.0)
│
├── Tier 0a: Length Penalty
│   └── len(text.split()) > 10 → NONE (confidence 0.0)
│
├── Tier 1: _fast_path_match(text)  [MODIFIED]
│   ├── Exact whole-string match → intent (confidence 1.0)
│   └── End-of-sentence match → intent (confidence 0.95)
│   └── If GOTO_SLIDE → _extract_slide_number(text)  [MODIFIED with ordinals]
│
├── Tier 2: Embedding cosine similarity (threshold 0.55)  [UNCHANGED]
│   └── If GOTO_SLIDE → _extract_slide_number(text)  [MODIFIED with ordinals]
│
└── Return NONE (confidence 0.0)
```

---

## 4. Testing Strategy

### 4.1 Existing Tests — Regression Verification

All 11 existing tests must continue to pass unchanged. Here's the explicit check for each:

| # | Test | Input | Why It Still Passes |
|---|---|---|---|
| 1 | `test_fast_path_next_slide` | `"next slide"` | Exact match → 1.0. 2 words, under length limit. |
| 2 | `test_fast_path_go_back` | `"go back"` | Exact match → 1.0. 2 words. |
| 3 | `test_fast_path_case_insensitive` | `"Next Slide"` | Normalized exact match → 1.0. 2 words. |
| 4 | `test_intent_classifier_exact_matches` | `"next slide"`, `"go back"` | Same as #1/#2. |
| 5 | `test_intent_classifier_semantic_matches` | `"Let's move on..."` (7 words), `"Could you show..."` (7 words) | Under 10-word limit, reaches embedding path, matches semantically. |
| 6 | `test_goto_slide_with_digit` | `"go to slide 3"` | 4 words. Not an exact fast-path match. Endswith: ends with `"slide 3"`, no canonical phrase. Falls to embedding → GOTO_SLIDE → extracts `3`. |
| 7 | `test_goto_slide_with_word_number` | `"jump to slide five"` | 4 words. Similar to #6. Embedding → GOTO_SLIDE → extracts `5` (cardinal). |
| 8 | `test_goto_slide_semantic` | `"can we skip to slide number 7"` | 7 words. Embedding → GOTO_SLIDE → extracts `7` (digit). |
| 9 | `test_next_point` | `"next point"` | Exact match → 1.0. |
| 10 | `test_prev_point` | `"previous bullet"` | Exact match → 1.0. |
| 11 | `test_intent_classifier_no_match` | `"This is a really great point about quarter three revenue"` (10 words), `""`, `"   "` | 10 words = NOT > 10, so length penalty doesn't fire. Falls through to embedding → NONE. Empty/whitespace → NONE via existing guard. |

### 4.2 New Test Cases — Tier 0a Length Penalty

- **`test_length_penalty_long_sentence`**
  - Input: `"I think we should probably move on to discuss the next slide in our presentation"`
    (15 words)
  - Expected: `intent == "NONE"`, `confidence == 0.0`
  - Verifies: >10 words triggers immediate rejection despite containing "next slide"

- **`test_length_penalty_boundary_11_words`**
  - Input: `"I really think we should go ahead and go to next slide"` (12 words)
  - Expected: `intent == "NONE"`, `confidence == 0.0`
  - Verifies: 12 words (above boundary) is rejected

- **`test_length_penalty_boundary_10_words_passes`**
  - Input: `"I think we should now go to the next slide"` (10 words)
  - Expected: `intent != "NONE"` (should match NEXT_SLIDE via endswith or embedding)
  - Verifies: Exactly 10 words is NOT rejected (the check is `> 10`, not `>= 10`)

### 4.3 New Test Cases — Tier 0b End-of-Sentence Bias

- **`test_endswith_next_slide`**
  - Input: `"please go to the next slide"`
  - Expected: `intent == "NEXT_SLIDE"`, `confidence == 0.95`
  - Verifies: End-of-sentence match with correct reduced confidence

- **`test_endswith_go_back`**
  - Input: `"can we go back"`
  - Expected: `intent == "PREV_SLIDE"`, `confidence == 0.95`
  - Verifies: End-of-sentence match for PREV_SLIDE

- **`test_endswith_continue`**
  - Input: `"alright let's continue"`
  - Expected: `intent == "NEXT_SLIDE"`, `confidence == 0.95`
  - Verifies: Single canonical word at end of sentence

- **`test_endswith_move_forward`**
  - Input: `"okay please move forward"`
  - Expected: `intent == "NEXT_SLIDE"`, `confidence == 0.95`
  - Verifies: Multi-word canonical phrase at end

### 4.4 New Test Cases — False Positive Rejection

- **`test_false_positive_mid_sentence`**
  - Input: `"the next slide shows our revenue"`
  - Expected: `intent == "NONE"`
  - Verifies: Command phrase at START of sentence followed by more words is rejected.
    "next slide" is at position 1–2, not at the end. Fast-path endswith won't match.
    Length is 6 words (under limit), so it reaches embedding, but the embedding model
    should score below 0.55 since the sentence is about "revenue", not a navigation command.

- **`test_false_positive_previous_slide_start`**
  - Input: `"previous slide had a great chart"`
  - Expected: `intent == "NONE"`
  - Verifies: "previous slide" at start, followed by non-command continuation.

- **`test_false_positive_go_back_mid_sentence`**
  - Input: `"if we go back to the data"`
  - Expected: `intent == "NONE"`
  - Verifies: "go back" in the middle of a sentence, followed by more words.

### 4.5 New Test Cases — Ordinal Number Extraction

- **`test_goto_slide_ordinal_first`**
  - Input: `"go to the first slide"`
  - Expected: `intent == "GOTO_SLIDE"`, `slide_number == 1`
  - Verifies: Basic ordinal extraction

- **`test_goto_slide_ordinal_third`**
  - Input: `"move to the third slide"`
  - Expected: `intent == "GOTO_SLIDE"`, `slide_number == 3`
  - Verifies: The exact failure case the user reported

- **`test_goto_slide_ordinal_tenth`**
  - Input: `"jump to the tenth slide"`
  - Expected: `intent == "GOTO_SLIDE"`, `slide_number == 10`
  - Verifies: Double-digit ordinal

- **`test_goto_slide_ordinal_twentieth`**
  - Input: `"show the twentieth slide"`
  - Expected: `intent == "GOTO_SLIDE"`, `slide_number == 20`
  - Verifies: Upper boundary of ordinal map

- **`test_goto_slide_ordinal_fifth`**
  - Input: `"can we skip to the fifth slide"`
  - Expected: `intent == "GOTO_SLIDE"`, `slide_number == 5`
  - Verifies: Ordinal inside a polite/natural sentence

### 4.6 New Test Cases — Word Boundary Safety (Substring Bug Fix)

- **`test_extract_eighteen_not_eight`**
  - Input: `"go to slide eighteen"`
  - Expected: `slide_number == 18` (not `8`)
  - Verifies: `\beight\b` regex does NOT match inside `"eighteen"`

- **`test_extract_fourteenth_not_four`**
  - Input: `"go to the fourteenth slide"`
  - Expected: `slide_number == 14` (not `4`)
  - Verifies: `\bfour\b` regex does NOT match inside `"fourteenth"`, and
    `\bfourteenth\b` correctly matches in the ordinal map

### 4.7 New Test Case — Ordinal + Cardinal Coexistence

- **`test_digit_takes_priority_over_ordinal`**
  - Input: `"go to slide 3"` (already exists as `test_goto_slide_with_digit`, but
    confirm digit wins if both could match)
  - This is already covered by existing test #6. No new test needed.

### 4.8 Summary — All New Tests

| # | Test Name | Category | Input | Expected |
|---|---|---|---|---|
| 1 | `test_length_penalty_long_sentence` | Tier 0a | 15-word sentence with "next slide" | NONE, 0.0 |
| 2 | `test_length_penalty_boundary_11_words` | Tier 0a | 12-word sentence | NONE, 0.0 |
| 3 | `test_length_penalty_boundary_10_words_passes` | Tier 0a | 10-word sentence ending with command | NOT NONE |
| 4 | `test_endswith_next_slide` | Tier 0b | `"please go to the next slide"` | NEXT_SLIDE, 0.95 |
| 5 | `test_endswith_go_back` | Tier 0b | `"can we go back"` | PREV_SLIDE, 0.95 |
| 6 | `test_endswith_continue` | Tier 0b | `"alright let's continue"` | NEXT_SLIDE, 0.95 |
| 7 | `test_endswith_move_forward` | Tier 0b | `"okay please move forward"` | NEXT_SLIDE, 0.95 |
| 8 | `test_false_positive_mid_sentence` | False positive | `"the next slide shows our revenue"` | NONE |
| 9 | `test_false_positive_previous_slide_start` | False positive | `"previous slide had a great chart"` | NONE |
| 10 | `test_false_positive_go_back_mid_sentence` | False positive | `"if we go back to the data"` | NONE |
| 11 | `test_goto_slide_ordinal_first` | Ordinal | `"go to the first slide"` | GOTO_SLIDE, slide 1 |
| 12 | `test_goto_slide_ordinal_third` | Ordinal | `"move to the third slide"` | GOTO_SLIDE, slide 3 |
| 13 | `test_goto_slide_ordinal_tenth` | Ordinal | `"jump to the tenth slide"` | GOTO_SLIDE, slide 10 |
| 14 | `test_goto_slide_ordinal_twentieth` | Ordinal | `"show the twentieth slide"` | GOTO_SLIDE, slide 20 |
| 15 | `test_goto_slide_ordinal_fifth` | Ordinal | `"can we skip to the fifth slide"` | GOTO_SLIDE, slide 5 |
| 16 | `test_extract_eighteen_not_eight` | Word boundary | `"go to slide eighteen"` | GOTO_SLIDE, slide 18 |
| 17 | `test_extract_fourteenth_not_four` | Word boundary | `"go to the fourteenth slide"` | GOTO_SLIDE, slide 14 |

**Total**: 11 existing + 17 new = **28 tests**.

---

## 5. Files Changed — Summary

| File | Action | Changes |
|---|---|---|
| `backend/intent_classifier.py` | **Modify** | (1) Add `ORDINAL_TO_NUM` dict (4 lines). (2) Update `_extract_slide_number` with ordinal scan + word-boundary regex fix (6 lines changed). (3) Add Tier 0a length penalty in `classify_intent` (4 lines). (4) Rewrite `_fast_path_match` with end-of-sentence scan (12 lines). Net: ~20 lines added. |
| `tests/test_intent_classifier.py` | **Modify** | Add 17 new test functions organized into 4 sections (length penalty, endswith, false positive, ordinal). Net: ~90 lines added. |

**No other files are modified.** The heuristics are purely within the intent classifier —
`app.py`, `vad_engine.py`, `transcriber.py`, and all frontend files are untouched.

---

## 6. Execution Order

1. **`backend/intent_classifier.py`** — Add `ORDINAL_TO_NUM` dict after `WORD_TO_NUM`.
2. **`backend/intent_classifier.py`** — Fix `_extract_slide_number` with word-boundary regex and ordinal scan.
3. **`backend/intent_classifier.py`** — Rewrite `_fast_path_match` with end-of-sentence bias.
4. **`backend/intent_classifier.py`** — Add Tier 0a length penalty to `classify_intent`.
5. **`tests/test_intent_classifier.py`** — Add all 17 new test cases.
6. **Run `pytest tests/test_intent_classifier.py -v`** — All 28 tests must pass.
7. **Manual smoke test** — Start server, speak ordinal commands and natural sentences, verify behavior.

---

## 7. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Embedding path matches false-positive sentences below 10 words (e.g., `"the next slide shows our revenue"` is 6 words) | False navigation trigger | The embedding model should score these below 0.55 because the full sentence context dilutes the command sense. If testing reveals scores ≥ 0.55, we can add a Tier 0c positional check on the embedding result. |
| End-of-sentence scan creates unexpected matches for short canonical words like `"continue"` | `"I'll continue the discussion"` → false NEXT_SLIDE | Won't happen: `"I'll continue the discussion"` ends with `"discussion"`, not `"continue"`. Only `"alright let's continue"` (ending with the word) would match. The endswith logic is inherently safe for this. |
| Ordinal map doesn't cover compound ordinals (`"twenty-first"`, `"thirty-second"`) | Slides 21+ can't be navigated by ordinal word | Acceptable limitation. Slides 21+ can use digit input (`"go to slide 25"`). Map can be extended later if user demand arises. |
| Word-boundary regex `\bfour\b` change could have subtle Unicode edge cases | Unlikely in English speech transcripts | Whisper outputs ASCII English text. No Unicode boundary issues expected. |
| `"second to last"` ordinal extraction returns `2` instead of relative position | Wrong slide navigation | Known limitation (documented in §2.5). Relative position commands ("second to last", "third from the end") are out of scope for Phase 4. |

---

## 8. Known Limitations (Out of Scope)

- **Compound ordinals** (`"twenty-first"` through `"ninety-ninth"`) — not supported, use digits.
- **Relative ordinals** (`"second to last"`, `"third from the end"`) — not supported.
- **Ordinals above twentieth** — not supported, use digits (`"go to slide 25"`).
- **Homophone ambiguity** (`"for"` vs `"four"`, `"won"` vs `"one"`) — Whisper's language model
  handles this contextually. We trust the transcriber output.
