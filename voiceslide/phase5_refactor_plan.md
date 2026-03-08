# Phase 5 Refactor — Universal Semantic Fallback

> **Status**: Blueprint for review — **no code will be written until approved.**
>
> **Motivation**: The Tier 1.5 Prefix Matcher (`_GOTO_CONTENT_PREFIXES` +
> `_content_prefix_match()`) forces users to memorize phrases like
> "show me the …" or "find the slide about …". This ruins the NLP experience.
> A user saying *"let's look at the CISC architecture"* should work just as
> well as *"show me the CISC architecture"* — without any prefix requirement.

---

## 0. Architectural Summary

### Before (Current — Rigid)

```
Speech → Transcribe → Intent Classifier (4 tiers, GOTO_CONTENT in Tier 1.5)
                                ↓
                    GOTO_CONTENT matched? ──→ context_search.search()
                    GOTO_SLIDE matched?   ──→ emit nav_command
                    NONE matched?         ──→ discard (dead end)
```

**Problem**: Only speech matching a hardcoded prefix (9 patterns) can reach
`context_search`. Everything else that returns `NONE` is silently discarded —
even if it's a perfectly valid content query.

### After (Proposed — Universal Fallback)

```
Speech → Transcribe → Intent Classifier (3 tiers, NO GOTO_CONTENT)
                                ↓
                    Intent recognized?  ──→ execute command (NEXT, PREV, GOTO_SLIDE…)
                    NONE returned?      ──→ context_search.search(raw text)
                                                   ↓
                                           Match ≥ 0.30? ──→ emit GOTO_CONTENT
                                           No match?     ──→ discard (truly non-navigational)
```

**Key insight**: The intent classifier handles *commands*. The context search
handles *topics*. They are two different cognitive categories and should be
evaluated in sequence, not mixed into the same classification tier.

---

## 1. Intent Classifier Cleanup (`intent_classifier.py`)

### 1.1 What Gets Removed

| Component | Lines | Reason |
|---|---|---|
| `"GOTO_CONTENT"` entry in `CANONICAL_INTENTS` | 49–54 | No longer a classifier intent; handled by fallback |
| `_GOTO_CONTENT_PREFIXES` tuple | 130–140 | The rigid prefix list — core of the problem |
| `_content_prefix_match()` function | 143–156 | Called from Tier 1.5 — entire tier eliminated |
| Tier 1.5 call block in `classify_intent()` | 229–233 | `content = _content_prefix_match(text)` + return |
| Tier 2 `GOTO_CONTENT` rejection block | 253–263 | No GOTO_CONTENT canonicals to match against |
| `GOTO_CONTENT` exemption in `_is_mid_sentence_false_positive()` | 161 | Remove `"GOTO_CONTENT"` from the tuple on line 161. `GOTO_SLIDE` exemption stays because GOTO_SLIDE uses `_extract_slide_number()` as its own guard |

### 1.2 What Gets Updated

| Component | Change |
|---|---|
| Module docstring (lines 1–9) | Revert from "Four-tier matching" to **"Three-tier matching"**. Remove the "1.5 Prefix" line entirely |
| `FAST_PATH_MAP` (lines 63–66) | No code change needed — it iterates `CANONICAL_INTENTS` dynamically, so removing the `GOTO_CONTENT` key automatically excludes it |

### 1.3 What Stays Unchanged

- **`CANONICAL_INTENTS`** keys: `NEXT_SLIDE`, `PREV_SLIDE`, `NEXT_POINT`, `PREV_POINT`, `GOTO_SLIDE`, `START_PRESENTATION`, `END_PRESENTATION` (7 intents)
- **`_fast_path_match()`** — exact match + endswith scan (Tier 1)
- **`_extract_slide_number()`** — digit/cardinal/ordinal extraction
- **`_is_mid_sentence_false_positive()`** — keeps `GOTO_SLIDE` exemption, removes `GOTO_CONTENT`
- **`_initialize_classifier()`** — computes embeddings for the 7 remaining intents
- **Tier 0a** (length penalty >10 words)
- **Tier 0b** (endswith bias at 0.95)
- **Tier 0c** (mid-sentence false positive guard)
- **Tier 2** embedding cosine similarity — now only matches the 7 command intents
- **`CONFIDENCE_THRESHOLD = 0.55`**

### 1.4 Post-Cleanup `classify_intent()` Flow

```
classify_intent(text)
  ├─ Empty/whitespace? → NONE
  ├─ Tier 0a: len(words) > 10? → NONE
  ├─ Tier 1: _fast_path_match(text) → exact/endswith match?
  │     └─ If GOTO_SLIDE: attach slide_number
  │     └─ Return match
  ├─ Tier 2: embedding cosine ≥ 0.55?
  │     ├─ Tier 0c: mid-sentence false positive? → NONE
  │     ├─ If GOTO_SLIDE: attach slide_number
  │     └─ Return match
  └─ Return NONE (← this is now the fallback trigger)
```

The classifier becomes a **pure command recognizer**. It never returns
`GOTO_CONTENT`. It either recognizes a command or returns `NONE`.

---

## 2. The Fallback Logic (`app.py`)

### 2.1 Current Logic in `_process_speech_buffer()` (lines 234–271)

```python
# 2. Classify intent
intent_result = classify_intent(transcribed_text)
command = intent_result.get("intent")

if command and command != "NONE":
    # ... cooldown check ...
    payload = {"action": command}
    if command == "GOTO_SLIDE":
        slide_num = intent_result.get("slide_number")
        if slide_num is not None:
            payload["slide_number"] = slide_num
        else:
            # GOTO_SLIDE without a number — try content search as fallback
            match = context_search.search(transcribed_text)
            if match:
                payload = {
                    "action": "GOTO_CONTENT",
                    "slide_number": match["slide_index"] + 1,
                }
            else:
                return
    elif command == "GOTO_CONTENT":
        match = context_search.search(transcribed_text)
        if match:
            payload["slide_number"] = match["slide_index"] + 1
        else:
            return
    socketio.emit("nav_command", payload)
```

### 2.2 New Logic — Step by Step

```python
# 2. Classify intent
intent_result = classify_intent(transcribed_text)
command = intent_result.get("intent")

if command and command != "NONE":
    # ── Step A+B: Recognized command — execute normally ──────────
    current_time = time.time()
    if current_time - _last_nav_command_time > NAV_COMMAND_COOLDOWN:
        _last_nav_command_time = current_time

        payload = {"action": command}
        if command == "GOTO_SLIDE":
            slide_num = intent_result.get("slide_number")
            if slide_num is not None:
                payload["slide_number"] = slide_num
            else:
                # GOTO_SLIDE without a number — try content search
                match = context_search.search(transcribed_text)
                if match:
                    payload = {
                        "action": "GOTO_CONTENT",
                        "slide_number": match["slide_index"] + 1,
                    }
                else:
                    return
        socketio.emit("nav_command", payload)
    else:
        logger.debug("Ignored nav_command %s due to cooldown.", command)
else:
    # ── Step C: Universal Fallback — classifier returned NONE ────
    match = context_search.search(transcribed_text)
    if match:
        # ── Step D: Content match found — emit GOTO_CONTENT ──────
        current_time = time.time()
        if current_time - _last_nav_command_time > NAV_COMMAND_COOLDOWN:
            _last_nav_command_time = current_time
            logger.info(
                "Fallback content match: '%s' → slide %d (score: %.2f)",
                transcribed_text, match["slide_index"] + 1, match["score"],
            )
            socketio.emit("nav_command", {
                "action": "GOTO_CONTENT",
                "slide_number": match["slide_index"] + 1,
            })
        else:
            logger.debug("Ignored fallback GOTO_CONTENT due to cooldown.")
    else:
        logger.debug("No intent or content match for: '%s'", transcribed_text)
```

### 2.3 Key Design Decisions

| Decision | Rationale |
|---|---|
| Fallback uses the same `NAV_COMMAND_COOLDOWN` | Prevents rapid-fire slide jumps from background conversation |
| Fallback sits in the `else` branch (command is `NONE`) | Clean separation — recognized commands never hit the fallback path |
| `context_search.search()` threshold stays at `0.30` | Already validated in Phase 5 v1. The threshold filters out non-navigational chatter (e.g., "that's an interesting point" scores well below 0.30 against slide content) |
| `GOTO_SLIDE` without a number still falls through to `context_search` | Preserves existing behavior — "go to the budget slide" matches GOTO_SLIDE but has no number, so content search resolves the target slide |
| The `elif command == "GOTO_CONTENT"` block is **deleted entirely** | The classifier will never return `GOTO_CONTENT`, so this branch is dead code |
| `GOTO_CONTENT` frontend handling in `app.js` stays as-is | The frontend doesn't care how the backend decided to emit the payload — it just receives `{action: "GOTO_CONTENT", slide_number: N}` and navigates |

### 2.4 What Stays Unchanged in `app.py`

- Lines 1–2: `import eventlet; eventlet.monkey_patch()` (NEVER TOUCH)
- Lines 48–61: `context_search` import + `_index_current_slides()` + startup call
- Lines 139–140: Re-index after PPTX upload
- Lines 164–165: Re-index after JSON save
- All WebSocket handlers (`audio_chunk`, `connect`, `disconnect`)
- All REST API routes (`/api/slides`, `/api/upload-pptx`, `/api/save-slides`)
- All page/static routes (`/`, `/upload`, `/css/`, `/js/`, `/static/images/`)
- VAD state machine logic

---

## 3. VRAM and Performance Justification

### 3.1 Model Inventory (4 GB VRAM Budget)

| Model | VRAM | Status |
|---|---|---|
| Silero VAD v4 (JIT) | ~2 MB | On CUDA — negligible |
| faster-whisper `base` (float16) | ~150 MB | On CUDA — singleton |
| all-MiniLM-L6-v2 | ~90 MB | On CUDA — singleton, shared |
| **Total** | **~242 MB** | Well within 4 GB |

### 3.2 Why This Approach Is Cheaper

**Previous (Tier 1.5 + GOTO_CONTENT in classifier)**:
- The 8 GOTO_CONTENT canonical phrases were included in the Tier 2 embedding corpus.
- Every voice command triggered a cosine similarity search across **all** canonicals,
  including the GOTO_CONTENT ones.
- GOTO_CONTENT canonicals actively caused false positives (vocabulary overlap), requiring
  an extra rejection pass at Tier 2.
- Net compute: classifier embedding search (always) + sometimes context search.

**Universal Fallback (proposed)**:
- Classifier corpus shrinks from ~37 phrases to ~29 phrases (8 GOTO_CONTENT removed).
- Cosine similarity search over fewer canonical embeddings → marginally faster.
- Context search only runs when the classifier returns `NONE` — it does **not** run when
  a command is successfully recognized.
- The same `all-MiniLM-L6-v2` model handles both classifier and context search. No
  additional model loading. No additional VRAM. The `encode()` singleton in
  `embeddings.py` is shared.
- The Tier 1.5 prefix matching code is deleted → fewer function calls per classification.

**Net effect**: Slightly fewer embedding comparisons in the classifier, and context
search is only invoked on `NONE` results instead of on explicit `GOTO_CONTENT` matches.
No new models, no new VRAM, no new dependencies.

### 3.3 Why Not a Local SLM?

A small language model (e.g., Phi-3-mini, TinyLlama) for intent understanding would:
- Require **1.5–3 GB additional VRAM** — would exceed the 4 GB budget.
- Add **200–800 ms latency** per inference — unacceptable for real-time voice navigation.
- Require a new dependency and model download pipeline.
- Be overkill — the two-stage approach (command classifier + content search) covers
  the same use cases with ~10–20 ms total latency using models already loaded.

---

## 4. Testing Strategy Updates

### 4.1 Tests to Delete (3 tests)

These tests assert that specific prefix phrases trigger `GOTO_CONTENT` at the
**classifier** level. Since the classifier will no longer return `GOTO_CONTENT`,
these tests become invalid.

| Test | File | Line | Reason |
|---|---|---|---|
| `test_goto_content_show_me` | `test_intent_classifier.py` | 259–263 | Asserts `classify_intent("show me the Harvard architecture")` returns `GOTO_CONTENT` with confidence `0.90`. After refactor, classifier returns `NONE`; content matching happens in `app.py` fallback. |
| `test_goto_content_revisit` | `test_intent_classifier.py` | 265–268 | Same — prefix match for "let's revisit the RISC features". |
| `test_goto_content_find` | `test_intent_classifier.py` | 270–273 | Same — prefix match for "find the slide about pipelining". |

### 4.2 Tests to Update (2 tests)

| Test | Current Assertion | New Assertion | Reason |
|---|---|---|---|
| `test_goto_slide_still_wins_with_number` (line 275) | Asserts GOTO_SLIDE for "go to slide 3" — framed as "must NOT become GOTO_CONTENT" | **Keep assertion, update docstring only**. GOTO_SLIDE is still correct; the regression gate framing ("still wins") is now misleading since there's no GOTO_CONTENT to compete with. Rename to: `test_goto_slide_basic_with_number` |
| `test_goto_slide_ordinal_still_wins` (line 281) | Same framing for "jump to the 4th slide" | **Keep assertion, update docstring only**. Rename to: `test_goto_slide_ordinal_basic` |

### 4.3 Tests to Keep Unchanged

All other 34 intent classifier tests remain valid:
- Fast-path exact matches (3 tests)
- Embedding-based semantic matches (2 tests)
- GOTO_SLIDE with number extraction (3 tests)
- NEXT_POINT / PREV_POINT (2 tests)
- Non-command speech → NONE (1 test)
- Tier 0a length penalty (3 tests)
- Tier 0b endswith bias (4 tests)
- False positive rejection (3 tests)
- Ordinal number extraction (5 tests)
- Word boundary safety (2 tests)
- Digit-ordinal extraction (6 tests)

### 4.4 New Tests — Classifier Level (`test_intent_classifier.py`)

These verify that natural content queries **correctly return `NONE`** from the
classifier (proving the classifier no longer intercepts them):

| Test Name | Input | Expected |
|---|---|---|
| `test_natural_query_returns_none_cisc` | `"let's look at the CISC architecture"` | `intent == "NONE"` |
| `test_natural_query_returns_none_harvard` | `"what about the Harvard stuff"` | `intent == "NONE"` |
| `test_natural_query_returns_none_show_me` | `"show me the Harvard architecture"` | `intent == "NONE"` |
| `test_natural_query_returns_none_revisit` | `"let's revisit the RISC features"` | `intent == "NONE"` |

The first two prove that conversational queries (no rigid prefix) reach NONE.
The last two prove that the **former prefix phrases** also reach NONE — they are
no longer special-cased.

### 4.5 New Tests — Integration Level (new file: `test_universal_fallback.py`)

These test the full `_process_speech_buffer()` fallback path end-to-end.
They will require mocking `transcribe_chunk`, `classify_intent`, and
`context_search.search`, plus capturing `socketio.emit` calls.

| Test Name | Setup | Assertion |
|---|---|---|
| `test_fallback_emits_goto_content` | `classify_intent` returns `NONE`; `context_search.search` returns `{"slide_index": 4, "score": 0.42, "matched_text": "..."}` | `socketio.emit` called with `{"action": "GOTO_CONTENT", "slide_number": 5}` |
| `test_fallback_no_match_no_emit` | `classify_intent` returns `NONE`; `context_search.search` returns `None` | `socketio.emit` NOT called |
| `test_recognized_command_skips_fallback` | `classify_intent` returns `NEXT_SLIDE`; `context_search.search` NOT called | `socketio.emit` called with `{"action": "NEXT_SLIDE"}` |
| `test_goto_slide_without_number_uses_context` | `classify_intent` returns `GOTO_SLIDE` (no slide_number); `context_search.search` returns match | `socketio.emit` called with `{"action": "GOTO_CONTENT", "slide_number": N}` |
| `test_fallback_respects_cooldown` | `classify_intent` returns `NONE`; `context_search.search` returns match; `_last_nav_command_time` is recent | `socketio.emit` NOT called |

### 4.6 Existing Context Search Tests (Unchanged)

All 8 tests in `test_context_search.py` remain valid and untouched. They test
`context_search.build_index()` and `context_search.search()` independently of
how the caller decides to invoke them.

---

## 5. Files Changed — Summary

| File | Action | Net Change |
|---|---|---|
| `backend/intent_classifier.py` | **Modify** | Remove ~35 lines (GOTO_CONTENT canonicals, prefix matcher, Tier 1.5 block, Tier 2 rejection). Update docstring. Update `_is_mid_sentence_false_positive` exemption tuple. |
| `backend/app.py` | **Modify** | Replace the `if command == "GOTO_CONTENT"` block (~8 lines) with the universal fallback `else` branch (~15 lines). Net: ~7 lines added. |
| `tests/test_intent_classifier.py` | **Modify** | Delete 3 prefix tests (~15 lines). Update 2 docstrings. Add 4 new NONE-assertion tests (~20 lines). Net: ~5 lines added. |
| `tests/test_universal_fallback.py` | **Create** | New integration test file (~80 lines). 5 tests covering the fallback path with mocks. |

**No changes to**: `context_search.py`, `embeddings.py`, `config.py`,
`transcriber.py`, `vad_engine.py`, `slide_loader.py`, `app.js`,
`audio-processor.js`, `index.html`, `upload.html`, any CSS files.

---

## 6. Execution Order

1. **`backend/intent_classifier.py`** — Remove all GOTO_CONTENT code, update docstring.
2. **`backend/app.py`** — Replace GOTO_CONTENT branch with universal fallback.
3. **`tests/test_intent_classifier.py`** — Delete prefix tests, update docstrings, add NONE tests.
4. **`tests/test_universal_fallback.py`** — Create integration tests with mocks.
5. **Run all tests** — Verify the full suite passes.

---

## 7. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Non-navigational chatter triggers content search | Unnecessary `context_search.search()` calls on every `NONE` result | The `0.30` threshold filters out irrelevant speech. General commentary (e.g., "that's a great point") scores well below 0.30 against slide-specific content. Additionally, `encode()` for a single short query takes ~2–5 ms — negligible overhead. |
| Random speech accidentally matches a slide | False positive navigation to wrong slide | The `SEARCH_THRESHOLD = 0.30` was validated in Phase 5 v1 with 8 passing tests. The threshold can be tuned upward (e.g., 0.35) if false positives are observed in practice, without any architectural change. |
| Tier 0a length penalty (>10 words) blocks long content queries | User says "I want to go back to the slide where we talked about CISC" (14 words) → rejected before reaching fallback | **This is a known limitation.** The length penalty exists to prevent false positives from presenter monologues. A future improvement could apply the length penalty only to the classifier path and bypass it for the fallback path, but that is out of scope for this refactor. The workaround: users naturally shorten content queries ("the CISC slide", "CISC architecture"). |
| `GOTO_SLIDE` without a number still uses content search | Existing behavior preserved | Not a new risk — this path already exists in the current code (lines 253–261). The refactor doesn't change it. |
