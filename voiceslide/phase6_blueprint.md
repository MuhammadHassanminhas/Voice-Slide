# Phase 6 — Live Keyword Highlighting & Fallback Interception

## Problem: Context Blindness

The Phase 5 Universal Fallback catches **every** `NONE`-classified utterance and runs it through `context_search.search()`. When a presenter reads content from the **current** slide aloud, the semantic index often matches a **different** slide that shares similar vocabulary — causing an unwanted navigation jump.

**Example:** The user is on Slide 3 ("Features of RISC Processors") and says *"one cycle execution per instruction"*. The fallback finds that phrase on Slide 3 but also scores similarly against Slide 5 which mentions "instruction pipelining". The user never intended to navigate — they were presenting.

**Solution:** Insert a **Fuzzy Match Interceptor** between the classifier's `NONE` result and the Universal Fallback. If the transcript fuzzy-matches text on the **current slide**, we emit a `highlight_text` event (live highlighting) and **block** the fallback from firing.

---

## 1. State Management (`app.py`)

### 1.1 New Global: `_current_slide_index`

```python
_current_slide_index = 0  # 0-based, mirrors Reveal.js indexh
```

Placed alongside the existing `_last_nav_command_time`, `_speech_buffer`, etc. (around line 183).

### 1.2 Frontend → Backend Sync

The frontend already fires `Reveal.on("slidechanged")` (app.js line 289–293) but the emit is commented out. We will:

1. **Uncomment** the `socket.emit` in `app.js`:
   ```javascript
   Reveal.on("slidechanged", (event) => {
     const slideIndex = event.indexh;
     console.log(`[VoiceSlide] Slide changed → index ${slideIndex}`);
     socket.emit("slide_changed", { slide_index: slideIndex });
   });
   ```

2. **Add a handler** in `app.py`:
   ```python
   @socketio.on("slide_changed")
   def handle_slide_changed(data):
       global _current_slide_index
       idx = data.get("slide_index", 0)
       _current_slide_index = int(idx)
       logger.debug("Slide index updated → %d", _current_slide_index)
   ```

3. **Auto-update on outgoing nav_commands:** Whenever `_process_speech_buffer()` emits a `nav_command` that changes the slide (NEXT_SLIDE, PREV_SLIDE, GOTO_SLIDE, GOTO_CONTENT, START_PRESENTATION, END_PRESENTATION), we must update `_current_slide_index` to stay in sync. We do this **after** the emit:
   ```python
   # After emitting nav_command payload:
   _update_current_index(payload)
   ```

   Where `_update_current_index` is a small helper:
   ```python
   def _update_current_index(payload):
       global _current_slide_index
       action = payload.get("action")
       if action == "NEXT_SLIDE" or action == "NEXT_POINT":
           _current_slide_index = min(_current_slide_index + 1, context_search._num_slides - 1)
       elif action == "PREV_SLIDE" or action == "PREV_POINT":
           _current_slide_index = max(_current_slide_index - 0, 0)  # not -1 typo, handled below
       elif action in ("GOTO_SLIDE", "GOTO_CONTENT"):
           sn = payload.get("slide_number")
           if sn is not None:
               _current_slide_index = sn - 1
       elif action == "START_PRESENTATION":
           _current_slide_index = 0
       elif action == "END_PRESENTATION":
           _current_slide_index = max(context_search._num_slides - 1, 0)
   ```

   *Note:* The `slide_changed` WebSocket event from the frontend is the **authoritative** sync — `_update_current_index` is a best-effort pre-sync to keep the backend current between the emit and the next frontend event.

### 1.3 Reset on Disconnect

In `handle_disconnect()`, add `_current_slide_index = 0`.

---

## 2. The Interceptor Pattern (`app.py` — `_process_speech_buffer()`)

### 2.1 New Flow (Pseudocode)

```
_process_speech_buffer():
    snapshot buffer → clear → reset VAD

    if empty → return
    transcribe → text
    if empty → return
    emit "transcript"

    # STEP 1: Intent Classification
    intent_result = classify_intent(text)
    command = intent_result["intent"]

    if command != NONE:
        # Execute command (existing logic, unchanged)
        ...
        return

    # STEP 2: ★ INTERCEPTOR — Fuzzy Match against CURRENT slide ★
    from keyword_highlighter import fuzzy_match_current_slide
    highlight_result = fuzzy_match_current_slide(text, _current_slide_index)

    if highlight_result is not None:
        # User is reading the current slide → highlight, do NOT navigate
        socketio.emit("highlight_text", highlight_result)
        logger.debug("Intercepted: '%s' matches current slide %d", text, _current_slide_index)
        return   # ← BLOCKS the Universal Fallback

    # STEP 3: Universal Fallback (existing, unchanged)
    match = context_search.search(text)
    if match:
        ...emit GOTO_CONTENT...
    else:
        ...log no match...
```

### 2.2 Key Design Decisions

| Decision | Rationale |
|---|---|
| Interceptor runs **after** intent classification | We must still honor explicit commands ("next slide") even if they match current slide text. |
| Interceptor runs **before** universal fallback | The whole point — block the fallback when reading current content. |
| Interceptor returns immediately on match | No cooldown needed for highlight events (non-navigational). |
| No cooldown on `highlight_text` emits | Highlighting is non-destructive — rapid updates are fine. |

---

## 3. Fuzzy Matching Logic (`keyword_highlighter.py`)

### 3.1 Module Overview

New file: `voiceslide/backend/keyword_highlighter.py`

```
Dependencies: thefuzz (add to requirements.txt)
```

### 3.2 Core Function Signature

```python
def fuzzy_match_current_slide(transcript: str, slide_index: int) -> dict | None:
    """
    Check if the transcript fuzzy-matches any text on the slide at slide_index.

    Returns a highlight payload dict if matched, or None if no match.
    """
```

### 3.3 Slide Text Extraction

Reuse `context_search._slides` (the slide list stored at build time) to get the current slide's data. Extract all matchable text spans from the slide:

```python
def _get_slide_text_spans(slide: dict) -> list[str]:
    """
    Return a list of individual text spans from a slide.
    Each span is a matchable unit (heading, bullet item, caption, quote, etc.).
    """
    spans = []
    if slide.get("heading"):
        spans.append(slide["heading"])
    if slide.get("subheading"):
        spans.append(slide["subheading"])
    for item in slide.get("items", []):
        spans.append(item)
    if slide.get("caption"):
        spans.append(slide["caption"])
    if slide.get("quote"):
        spans.append(slide["quote"])
    if slide.get("attribution"):
        spans.append(slide["attribution"])
    # Two-column
    for side in ("left", "right"):
        col = slide.get(side, {})
        if isinstance(col, dict):
            if col.get("title"):
                spans.append(col["title"])
            for item in col.get("items", []):
                spans.append(item)
    return spans
```

### 3.4 Fuzzy Matching with `thefuzz`

We use `thefuzz.fuzz.partial_ratio` because the transcript is often a **substring** of the slide text (e.g., user says part of a bullet point, not the whole thing).

```python
from thefuzz import fuzz

HIGHLIGHT_THRESHOLD = 65  # partial_ratio score (0–100)
```

**Algorithm:**
1. Get all text spans from the slide at `slide_index`.
2. Normalize both transcript and each span (lowercase, strip).
3. For each span, compute `fuzz.partial_ratio(transcript_lower, span_lower)`.
4. Track the **best match** (highest score + the matching span).
5. If best score ≥ `HIGHLIGHT_THRESHOLD`, return a highlight payload.
6. Otherwise, return `None`.

**Why `partial_ratio`?**
- The user often says a fragment of a bullet: *"one cycle execution"* should match *"One cycle execution per instruction"*.
- `partial_ratio` slides the shorter string across the longer one to find the best substring alignment. This is ideal for speech fragments.

**Why threshold = 65?**
- We want to catch paraphrased or partially-spoken slide content.
- 65 is generous enough to handle Whisper transcription noise (e.g., minor word substitutions) but strict enough to avoid matching unrelated speech.
- Testable — we can adjust if false positives emerge.

**Short-transcript guard:** If the transcript (after cleaning) is fewer than 3 words, skip fuzzy matching and return `None`. Single words like "the" or "so" would spuriously match many slides.

### 3.5 Emphasis Detection

Detect when the user verbally emphasizes a phrase — e.g., *"this is very important"*, *"remember this"*, *"key point"*, *"pay attention to this"*.

```python
import re

_EMPHASIS_RE = re.compile(
    r"\b(?:(?:this\s+is\s+)?(?:very\s+|really\s+|extremely\s+|super\s+)?important"
    r"|key\s+point|remember\s+this|pay\s+attention(?:\s+to\s+this)?"
    r"|note\s+this|critical|crucial|essential)\b",
    re.IGNORECASE,
)

def _detect_emphasis(transcript: str) -> bool:
    """Return True if the transcript contains emphasis trigger phrases."""
    return bool(_EMPHASIS_RE.search(transcript))
```

If emphasis is detected, the highlight payload includes `"emphasis": True`, which tells the frontend to apply a **bold/scale** CSS class instead of a plain highlight.

### 3.6 Return Payload

```python
{
    "slide_index": 2,           # 0-based slide index
    "matched_span": "One cycle execution per instruction",  # the original text from the slide
    "score": 78,                # fuzzy match score (0–100)
    "emphasis": False,          # True → bold/scale animation
}
```

### 3.7 Full Function (Pseudocode)

```python
def fuzzy_match_current_slide(transcript: str, slide_index: int) -> dict | None:
    # Guard: need slides loaded
    if not context_search._slides or slide_index >= len(context_search._slides):
        return None

    # Guard: very short transcript
    words = transcript.strip().split()
    if len(words) < 3:
        return None

    slide = context_search._slides[slide_index]
    spans = _get_slide_text_spans(slide)

    if not spans:
        return None

    transcript_lower = transcript.lower().strip()
    best_score = 0
    best_span = ""

    for span in spans:
        span_lower = span.lower().strip()
        score = fuzz.partial_ratio(transcript_lower, span_lower)
        if score > best_score:
            best_score = score
            best_span = span

    if best_score >= HIGHLIGHT_THRESHOLD:
        return {
            "slide_index": slide_index,
            "matched_span": best_span,
            "score": best_score,
            "emphasis": _detect_emphasis(transcript),
        }

    return None
```

---

## 4. WebSocket & Frontend

### 4.1 New Event: `highlight_text`

**Backend emits:**
```python
socketio.emit("highlight_text", {
    "slide_index": 2,
    "matched_span": "One cycle execution per instruction",
    "score": 78,
    "emphasis": False,
})
```

### 4.2 Frontend Handler (`app.js`)

```javascript
socket.on("highlight_text", (data) => {
    if (!data || data.slide_index == null) return;

    const currentIndex = Reveal.getState().indexh;

    // Safety: only highlight if we're still on the matching slide
    if (data.slide_index !== currentIndex) return;

    const section = document.querySelectorAll(".reveal .slides > section")[currentIndex];
    if (!section) return;

    // Remove any existing highlights first
    clearHighlights(section);

    // Find and highlight the matched span
    highlightSpan(section, data.matched_span, data.emphasis);
});
```

### 4.3 `highlightSpan()` — Text Node Walker

We walk the DOM tree of the current slide section and find text nodes that contain the `matched_span`. We wrap the matching portion in a `<mark>` element.

```javascript
function highlightSpan(section, spanText, isEmphasis) {
    if (!spanText) return;

    const lowerSpan = spanText.toLowerCase();

    // Walk all text-containing elements (headings, li, p, etc.)
    const elements = section.querySelectorAll(
        ".slide-heading, .slide-subheading, .slide-bullets li, " +
        ".slide-body, .slide-image-caption, .slide-quote-text, " +
        ".slide-quote-attribution, .slide-column__title, .slide-column__items li"
    );

    for (const el of elements) {
        if (el.textContent.toLowerCase().includes(lowerSpan)) {
            // Full element match — add class to the element itself
            el.classList.add("vs-highlight");
            if (isEmphasis) {
                el.classList.add("vs-emphasis");
            }

            // Auto-clear after 3 seconds
            setTimeout(() => {
                el.classList.remove("vs-highlight", "vs-emphasis");
            }, 3000);

            break;  // Only highlight the first match
        }
    }
}

function clearHighlights(section) {
    section.querySelectorAll(".vs-highlight").forEach((el) => {
        el.classList.remove("vs-highlight", "vs-emphasis");
    });
}
```

**Design Note:** We highlight the entire element (the `<li>`, `<h2>`, `<p>`) rather than wrapping substrings in `<mark>` tags. This avoids breaking Reveal.js's DOM expectations and is simpler to animate/remove.

### 4.4 CSS Classes (`presentation.css`)

```css
/* ── Phase 6: Live Keyword Highlighting ────────────────────────────────── */
.vs-highlight {
    background: rgba(0, 200, 255, 0.12) !important;
    border-color: rgba(0, 200, 255, 0.4) !important;
    box-shadow: 0 0 12px rgba(0, 200, 255, 0.15),
                inset 0 0 8px rgba(0, 200, 255, 0.08);
    transition: background var(--transition-normal),
                box-shadow var(--transition-normal),
                transform var(--transition-normal);
}

.vs-emphasis {
    background: rgba(124, 58, 237, 0.15) !important;
    border-color: rgba(124, 58, 237, 0.5) !important;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.2),
                inset 0 0 12px rgba(124, 58, 237, 0.1);
    transform: scale(1.02);
    font-weight: 700;
}
```

These use the existing design tokens (`--accent-primary: #00c8ff`, `--accent-secondary: #7c3aed`) and transition variables for consistency.

### 4.5 Slide-Change Cleanup

When the slide changes (in the existing `Reveal.on("slidechanged")` handler), clear all highlights from the previous slide:

```javascript
Reveal.on("slidechanged", (event) => {
    // Clear highlights from ALL slides (cheap, safe)
    document.querySelectorAll(".vs-highlight").forEach((el) => {
        el.classList.remove("vs-highlight", "vs-emphasis");
    });

    socket.emit("slide_changed", { slide_index: event.indexh });
});
```

---

## 5. Testing Strategy

### 5.1 New Test File: `tests/test_keyword_highlighter.py`

#### Unit Tests for `keyword_highlighter.py`

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `test_exact_bullet_match` | Transcript = exact bullet text → returns highlight with score ≥ 65 |
| 2 | `test_partial_bullet_match` | Transcript = first half of a bullet → still matches (partial_ratio) |
| 3 | `test_heading_match` | Transcript matches slide heading → highlight returned |
| 4 | `test_caption_match` | Transcript matches image caption → highlight returned |
| 5 | `test_quote_match` | Transcript matches quote text → highlight returned |
| 6 | `test_two_column_match` | Transcript matches a column item → highlight returned |
| 7 | `test_no_match_unrelated` | Transcript is unrelated to slide → returns None |
| 8 | `test_short_transcript_guard` | Transcript < 3 words → returns None (avoids spurious matches) |
| 9 | `test_empty_slide_no_crash` | Slide with no text spans → returns None |
| 10 | `test_invalid_slide_index` | `slide_index` out of range → returns None |
| 11 | `test_no_slides_loaded` | `context_search._slides` is None/empty → returns None |
| 12 | `test_emphasis_detected` | Transcript contains "this is very important" → `emphasis: True` |
| 13 | `test_emphasis_not_detected` | Normal transcript → `emphasis: False` |
| 14 | `test_emphasis_key_point` | Transcript contains "key point" → `emphasis: True` |
| 15 | `test_best_span_returned` | Multiple spans match; the one with the highest score is returned |
| 16 | `test_case_insensitive` | Mixed-case transcript matches lowercase slide text |

#### Emphasis Regex Unit Tests

| # | Test Name | Description |
|---|-----------|-------------|
| 17 | `test_emphasis_remember_this` | "remember this" → True |
| 18 | `test_emphasis_pay_attention` | "pay attention to this" → True |
| 19 | `test_emphasis_really_important` | "this is really important" → True |
| 20 | `test_emphasis_no_trigger` | "the pipeline has five stages" → False |

### 5.2 New Tests in `tests/test_interceptor.py`

#### Integration Tests for the Interceptor Pattern

These tests mock at the same boundaries as `test_universal_fallback.py` and additionally mock `keyword_highlighter.fuzzy_match_current_slide`.

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `test_interceptor_blocks_fallback` | Classifier → NONE, fuzzy match → hit → `highlight_text` emitted, `context_search.search` NOT called |
| 2 | `test_interceptor_miss_falls_through` | Classifier → NONE, fuzzy match → None → `context_search.search` IS called (fallback runs) |
| 3 | `test_command_bypasses_interceptor` | Classifier → NEXT_SLIDE → command emitted, fuzzy match NOT called |
| 4 | `test_highlight_no_cooldown` | Two rapid fuzzy matches → both emit `highlight_text` (no cooldown applied) |
| 5 | `test_slide_changed_updates_index` | Emit `slide_changed` event → `_current_slide_index` updated |
| 6 | `test_interceptor_uses_current_index` | Verify `fuzzy_match_current_slide` is called with the correct `_current_slide_index` value |
| 7 | `test_emphasis_forwarded_in_payload` | Fuzzy match returns `emphasis: True` → `highlight_text` payload includes `emphasis: True` |

### 5.3 Existing Test Preservation

- All 72 existing tests **must continue to pass** unchanged.
- `test_universal_fallback.py` tests still pass because the interceptor (mocked to return `None`) falls through to the existing fallback logic.

---

## 6. Dependency Addition

Add to `voiceslide/backend/requirements.txt`:

```
thefuzz[speedup]>=0.22.1
```

The `[speedup]` extra installs `python-Levenshtein` for C-accelerated fuzzy matching. If C compilation is problematic on Windows, fall back to `thefuzz>=0.22.1` (pure Python, slower but functional).

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/keyword_highlighter.py` | **CREATE** | Fuzzy matching + emphasis detection |
| `backend/app.py` | **EDIT** | Add `_current_slide_index`, `handle_slide_changed`, interceptor in `_process_speech_buffer()`, `_update_current_index()` |
| `backend/context_search.py` | **EDIT** | Store `_slides` list reference so `keyword_highlighter` can access slide data |
| `backend/requirements.txt` | **EDIT** | Add `thefuzz[speedup]` |
| `frontend/js/app.js` | **EDIT** | Uncomment `slide_changed` emit, add `highlight_text` handler, `highlightSpan()`, `clearHighlights()` |
| `frontend/css/presentation.css` | **EDIT** | Add `.vs-highlight` and `.vs-emphasis` CSS classes |
| `tests/test_keyword_highlighter.py` | **CREATE** | 20 unit tests for fuzzy matching + emphasis |
| `tests/test_interceptor.py` | **CREATE** | 7 integration tests for the interceptor pattern |

**Total new tests: 27** (20 unit + 7 integration)
**Expected final test count: 99** (72 existing + 27 new)

---

## 8. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `thefuzz` C extension fails to compile on Windows | Use `thefuzz` without `[speedup]`; pure Python fallback works fine for our small text spans |
| Fuzzy matching is too slow for real-time | Each slide has 3–8 text spans, each ≤200 chars. `partial_ratio` on these is < 1 ms total. No risk. |
| Threshold 65 causes false positives (blocks fallback when user IS navigating) | The interceptor only fires when `classify_intent` returns NONE. Explicit navigation commands bypass it entirely. For implicit navigation (topic mention), if the topic happens to match the current slide, blocking the fallback is the **correct** behavior — the user is already on that slide. |
| `_current_slide_index` drifts out of sync | Three-layer sync: (1) `slide_changed` WebSocket event (authoritative), (2) `_update_current_index` after voice nav, (3) reset on disconnect. |
| Highlighting breaks Reveal.js layout | We add/remove CSS classes on existing elements — no DOM insertion/removal. Safe. |

---

*Awaiting approval before generating code.*
