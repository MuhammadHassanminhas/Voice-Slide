# Phase 4.1 — Implementation Plan: Smart VAD & NLP Heuristics

> **Status**: Plan for review. No code will be written until approved.
>
> **Objective**: Eliminate false negatives (chopped words from blind 250ms chunking) and false positives ("next slide" mid-sentence triggering navigation) by introducing Silero VAD and NLP heuristics.

---

## 0. Current Architecture Snapshot (What We're Changing)

Before detailing the plan, here's a precise summary of the live pipeline so every change is grounded in reality.

| Component | File | Current Behavior |
|---|---|---|
| **Audio Worklet** | `frontend/js/audio-processor.js` | Buffers 4000 Float32 samples (250ms @ 16kHz) → posts `Float32Array` to main thread. |
| **Frontend Socket** | `frontend/js/app.js` (line 331) | `socket.emit("audio_chunk", event.data.buffer)` — sends the raw `ArrayBuffer` (binary). |
| **Backend Handler** | `backend/app.py` (lines 152–215) | Rolling buffer: appends each chunk to `_audio_buffer` (bytearray). Trims FIFO at `MAX_BUFFER_BYTES` (192 KB / 3s). Transcribes when buffer exceeds `MIN_BUFFER_BYTES` (64 KB / 1s). Runs intent classification on **every** transcription. Clears buffer on nav command. |
| **Transcriber** | `backend/transcriber.py` (line 67) | `np.frombuffer(audio_bytes, dtype=np.float32)` → `model.transcribe(array)`. Audio is **PCM float32, 16kHz, mono**. |
| **Intent Classifier** | `backend/intent_classifier.py` | Tier 1: O(1) dict lookup on exact phrases. Tier 2: cosine similarity (threshold `0.55`). No sentence-structure awareness. |

### Audio Data Type Clarification

> [!IMPORTANT]
> The frontend AudioWorklet sends **Float32** samples (range `[-1.0, 1.0]`). The transcriber already interprets them as `np.float32`. The original Phase 4 plan draft mentioned `np.int16` as an example — **this is incorrect for our pipeline**. The `vad_engine.py` must use `np.float32` consistently. Silero VAD natively expects float32 audio normalized to `[-1, 1]` at 16kHz, so **no sample-format conversion is needed** — only `bytes → np.float32 → torch.Tensor`.

### Chunk Geometry

| Property | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Bit depth | 32-bit float (4 bytes/sample) |
| Chunk duration | 250 ms |
| Samples per chunk | 4,000 |
| Bytes per chunk | 16,000 (4000 × 4) |

Silero VAD accepts chunks between 30ms–2000ms at 16kHz. Our 250ms chunks (4000 samples) are within the valid input range — **no sub-windowing required**.

---

## 1. Dependencies

### File: `backend/requirements.txt`

**Add only**:
```
omegaconf>=2.3.0
```

**Do NOT add** `torch` or `torchaudio` — they are already installed manually for CUDA 12.1 and must not be overwritten by pip's CPU-only default resolution.

---

## 2. New Module: `backend/vad_engine.py`

### Purpose
Encapsulate Silero VAD model loading and per-chunk speech confidence scoring. Stateless per-call (the state machine lives in `app.py`).

### Class: `VADEngine`

```
VADEngine
├── __init__(threshold: float = 0.5)
│   ├── Lazy-loads Silero VAD v5 model via torch.hub.load("snakers4/silero-vad")
│   ├── Singleton pattern (module-level _instance, like transcriber.py)
│   ├── Stores self.model, self.threshold
│   └── Calls model.reset_states() once on init
│
├── get_speech_confidence(audio_bytes: bytes) -> float
│   ├── Guard: if len(audio_bytes) == 0 → return 0.0
│   ├── Convert: np.frombuffer(audio_bytes, dtype=np.float32)
│   ├── Guard: if array has NaN/Inf → log warning, return 0.0
│   ├── Convert: torch.from_numpy(array)
│   ├── Run: self.model(tensor, 16000)  → returns float confidence [0.0, 1.0]
│   └── Return the confidence score
│
├── is_speech(audio_bytes: bytes) -> bool
│   └── return self.get_speech_confidence(audio_bytes) >= self.threshold
│
└── reset() -> None
    └── self.model.reset_states()  (call between utterances)
```

### Design Decisions

1. **Why `bytes` input, not `Tensor`?** — `app.py` receives raw bytes from the WebSocket. Keeping the conversion inside `vad_engine.py` isolates the data-type boundary to a single module.

2. **Why expose both `get_speech_confidence()` and `is_speech()`?** — `app.py` needs the raw float for logging/debugging, but also a simple boolean for the state machine branching.

3. **Why `reset()` exists** — Silero VAD is a stateful LSTM. After an utterance ends and we fire transcription, we must reset the internal hidden states so the next utterance starts clean. Without this, confidence scores bleed across utterances.

4. **Error handling** — Any exception during model inference returns `0.0` (treat as silence) and logs the error. This prevents a VAD glitch from crashing the entire audio pipeline.

### Singleton Access

```python
# Module-level factory (mirrors transcriber.py pattern)
_engine: VADEngine | None = None

def get_vad_engine() -> VADEngine:
    global _engine
    if _engine is None:
        _engine = VADEngine()
    return _engine
```

---

## 3. Backend Refactor: `backend/app.py`

### What Gets Removed

| Symbol | Line(s) | Reason |
|---|---|---|
| `MAX_BUFFER_BYTES` | 153 | Replaced by VAD-driven accumulation |
| `MIN_BUFFER_BYTES` | 154 | No longer needed — VAD decides when to transcribe |
| Rolling buffer append + FIFO trim logic | 178–186 | Replaced by speech-only accumulation |
| "Transcribe every tick" logic | 188–189 | Replaced by end-of-speech trigger |

### What Gets Added

#### New Imports
```python
from vad_engine import get_vad_engine
```

#### New State Variables (module-level, same scope as current `_audio_buffer`)

| Variable | Type | Purpose |
|---|---|---|
| `_speech_buffer` | `bytearray` | Accumulates audio **only during active speech**. Replaces `_audio_buffer`. |
| `_silence_counter` | `int` | Counts consecutive silent chunks after speech. |
| `_is_speaking` | `bool` | Tracks whether we are inside a speech utterance. |
| `SILENCE_CHUNKS_THRESHOLD` | `int` (constant = `2`) | Number of consecutive silent chunks (2 × 250ms = 500ms) required to declare end-of-speech. |
| `MAX_SPEECH_DURATION_BYTES` | `int` (constant = `480000`) | Safety cap: 30 seconds of audio (16000 × 4 × 30). If the speaker never pauses, force-transcribe to prevent unbounded memory growth. |

#### Updated `handle_audio_chunk(data)` — VAD-Driven Event Loop

```
receive audio_chunk (bytes)
│
├── validate: isinstance(data, bytes), len > 0
│
├── vad = get_vad_engine()
├── confidence = vad.get_speech_confidence(data)
├── log: "VAD confidence: {confidence:.2f}"
│
├── IF confidence >= 0.5 (SPEECH DETECTED):
│   ├── _is_speaking = True
│   ├── _silence_counter = 0
│   └── _speech_buffer.extend(data)
│
├── ELIF _is_speaking is True (SILENCE AFTER SPEECH):
│   ├── _speech_buffer.extend(data)          ← trailing padding for complete words
│   ├── _silence_counter += 1
│   │
│   └── IF _silence_counter >= SILENCE_CHUNKS_THRESHOLD:
│       ├── ══════ FIRE TRANSCRIPTION ══════
│       ├── log: "Speech End detected, transcribing..."
│       ├── result = transcribe_chunk(bytes(_speech_buffer))
│       ├── emit("transcript", result)
│       ├── Run classify_intent(result["text"])
│       ├── Emit nav_command if intent != NONE (with cooldown check)
│       ├── ── RESET STATE ──
│       ├── _speech_buffer.clear()
│       ├── _silence_counter = 0
│       ├── _is_speaking = False
│       └── vad.reset()                      ← reset LSTM hidden states
│
├── ELIF _is_speaking is False (PURE SILENCE):
│   └── discard chunk (do nothing)
│
└── SAFETY: if len(_speech_buffer) > MAX_SPEECH_DURATION_BYTES:
    ├── Force-transcribe current buffer
    ├── Reset all state
    └── log warning: "Speech buffer exceeded max duration, force-transcribing"
```

#### What Stays Unchanged

- `_last_nav_command_time` and `NAV_COMMAND_COOLDOWN` — the cooldown mechanism is still needed.
- The `nav_command` emission logic and payload structure.
- All REST API routes (slides, upload, save).
- The `connect`/`disconnect` handlers.

### Timing Analysis

| Scenario | Old Behavior | New Behavior |
|---|---|---|
| Speaker says "Next slide" (0.6s) then pauses | Buffer accumulates 250ms ticks, transcribes at 1s, may chop the word "slide" if the buffer resets mid-word | VAD detects speech start → accumulates 0.6s → detects 500ms silence → transcribes the complete utterance (0.6s + 0.5s padding = 1.1s of audio) |
| Speaker says "The next slide shows revenue" (2.5s continuous) | Transcribes multiple times during speech, may trigger "next slide" from a partial buffer | VAD keeps `_is_speaking = True` for the full 2.5s → transcribes once after pause → full sentence goes to classifier → heuristics reject it (see §4) |
| Speaker is silent for 30 seconds | Rolling buffer fills with silence, transcribes gibberish | `_is_speaking` stays False → all chunks discarded → zero transcriptions |

---

## 4. NLP Heuristics: `backend/intent_classifier.py`

### Where Heuristics Are Inserted

The heuristics form a new **Tier 0** that runs **before** both the fast-path and embedding lookup. This is a pre-filter that rejects text that is structurally unlikely to be a voice command.

```
classify_intent(text)
│
├── Tier 0: NLP Heuristics (NEW)
│   ├── Heuristic 1: Length Penalty
│   └── Heuristic 2: End-of-Sentence Bias (modifies fast-path)
│
├── Tier 1: Fast-path exact match (EXISTING, modified)
├── Tier 2: Embedding cosine similarity (EXISTING, unchanged)
└── Return NONE
```

### Heuristic 1: Length Penalty

**Rule**: If `len(text.split()) > 10`, return `{"intent": "NONE", "confidence": 0.0}` immediately.

**Rationale**: Real voice commands are short imperative phrases (2–6 words). Transcripts longer than 10 words are natural speech, not commands. This alone eliminates a large class of false positives where a command phrase appears inside a longer sentence.

**Implementation location**: Top of `classify_intent()`, right after the empty-string guard (line 126).

```python
# ── Tier 0a: Length Penalty ──
words = text.split()
if len(words) > 10:
    logger.debug("Length penalty: '%s' has %d words, skipping.", text, len(words))
    return {"intent": "NONE", "confidence": 0.0}
```

### Heuristic 2: End-of-Sentence Bias

**Problem**: The current fast-path matches **any** text that exactly equals a canonical phrase. But with VAD, the transcriber now receives full utterances. So "the next slide shows our revenue" would be transcribed as a whole, and while it won't exact-match (it's longer), the embedding path might still match with high similarity.

**Rule**: For both the fast-path and the embedding path, when a command phrase is found within the text, validate its **position**:

- ✅ **Valid** — phrase IS the entire text: `"next slide"` → trigger
- ✅ **Valid** — phrase is at the END of the text: `"please go to the next slide"` → trigger
- ❌ **Invalid** — phrase is in the MIDDLE or START followed by more words: `"the next slide shows revenue"` → reject

**Implementation approach**: Modify the fast-path to check `text.strip().lower().endswith(phrase)` or `text.strip().lower() == phrase` instead of only checking for exact equality. Then wrap the embedding path with a similar positional check.

**Detailed logic for the fast-path replacement**:

```python
def _fast_path_match(text: str) -> dict | None:
    normalized = text.strip().lower()

    # 1. Exact whole-string match (existing behavior)
    intent = FAST_PATH_MAP.get(normalized)
    if intent:
        return {"intent": intent, "confidence": 1.0}

    # 2. End-of-sentence match (NEW)
    #    Check if text ENDS with any canonical phrase.
    #    Only match if the phrase is preceded by a word boundary (space or start-of-string).
    for phrase, intent in FAST_PATH_MAP.items():
        if normalized.endswith(phrase) and normalized != phrase:
            # Ensure it's a word boundary, not a substring
            prefix = normalized[: -len(phrase)]
            if prefix == "" or prefix.endswith(" "):
                return {"intent": intent, "confidence": 0.95}

    return None
```

**Confidence note**: End-of-sentence matches get `0.95` instead of `1.0` to distinguish them from exact matches in logs.

**For the embedding path**: No structural change is needed. The length penalty (Heuristic 1) already filters out long sentences. Short sentences (≤10 words) that semantically match a command and where the command sense is dominant will still correctly trigger. The embedding model inherently gives lower scores to sentences where the command phrase is buried in unrelated context.

### Impact on Existing Tests

| Test | Expected Outcome After Changes |
|---|---|
| `test_fast_path_next_slide` ("next slide") | ✅ Still passes — exact whole-string match |
| `test_fast_path_go_back` ("go back") | ✅ Still passes — exact whole-string match |
| `test_fast_path_case_insensitive` ("Next Slide") | ✅ Still passes — normalized exact match |
| `test_intent_classifier_semantic_matches` ("Let's move on to the next one") | ✅ Still passes — 7 words (under limit), embedding match |
| `test_intent_classifier_semantic_matches` ("Could you show the previous slide again") | ✅ Still passes — 7 words, embedding match, ends with command-related words |
| `test_intent_classifier_no_match` (long sentence) | ✅ Still passes — "This is a really great point about quarter three revenue" is 10 words, right at the boundary. Need to verify. If it's exactly 10 words, the `> 10` check won't filter it, but it should still return NONE from the embedding path since it's not a command. |
| `test_goto_slide_*` | ✅ Still passes — "go to slide 3" is 4 words, exact fast-path match |

### New Test Cases to Add

| Test | Input | Expected |
|---|---|---|
| False positive: mid-sentence | `"The next slide shows our revenue"` | `NONE` |
| False positive: start-of-sentence | `"Previous slide was boring"` | `NONE` |
| True positive: end-of-sentence | `"Please go to the next slide"` | `NEXT_SLIDE` (confidence 0.95) |
| True positive: end-of-sentence (prev) | `"Can we go back"` | `PREV_SLIDE` (confidence 0.95) |
| Length penalty | `"I think we should probably move on to discuss the next slide in our presentation"` | `NONE` (15 words) |
| Short non-command | `"Great quarter everyone"` | `NONE` |

---

## 5. Files Changed — Summary

| File | Action | Scope |
|---|---|---|
| `backend/requirements.txt` | **Modify** | Add `omegaconf>=2.3.0` (1 line) |
| `backend/vad_engine.py` | **Create** | ~60 lines. `VADEngine` class + singleton factory. |
| `backend/app.py` | **Modify** | Remove rolling buffer constants/logic (~15 lines). Add VAD import + state variables + new event loop (~40 lines). Net change: ~25 lines added. |
| `backend/intent_classifier.py` | **Modify** | Add length penalty guard (~4 lines). Rewrite `_fast_path_match` to support end-of-sentence matching (~12 lines). |
| `tests/test_intent_classifier.py` | **Modify** | Add ~6 new test cases for heuristics. |

---

## 6. Execution Order

1. **`backend/requirements.txt`** — Add `omegaconf`. Install it (`pip install omegaconf`).
2. **`backend/vad_engine.py`** — Create the module. Verify model loads (`python -c "from vad_engine import get_vad_engine; get_vad_engine()"`).
3. **`backend/intent_classifier.py`** — Add heuristics (Tier 0). Run existing + new tests to confirm no regressions.
4. **`backend/app.py`** — Replace rolling buffer with VAD event loop.
5. **`tests/test_intent_classifier.py`** — Add false-positive/false-negative test cases.
6. **End-to-end verification** — Start server, speak commands, verify behavior.

---

## 7. Verification Plan

### Automated Checks

1. **Model Load**: `python -c "from vad_engine import get_vad_engine; e = get_vad_engine(); print('VAD loaded')"` — must complete without error.
2. **Unit Tests**: `pytest tests/test_intent_classifier.py -v` — all existing + new tests pass.
3. **Build Check**: Server starts without import errors.

### Manual Integration Tests

| # | Test | Action | Expected Log / Behavior |
|---|---|---|---|
| 1 | **VAD speech detection** | Say a sentence, pause for 1 second | `app.py` logs `"Speech End detected, transcribing..."` exactly **once** after the pause |
| 2 | **Silence suppression** | Remain silent for 10 seconds | Zero transcript emissions, zero `transcribe_chunk` calls |
| 3 | **Complete utterance** | Say "next slide" naturally | Full phrase transcribed (no chopped "next sli-"), `NEXT_SLIDE` triggers |
| 4 | **False positive rejection** | Say "The previous slide was boring" | Transcript appears in subtitle bar, but **no** `nav_command` emitted |
| 5 | **True positive (end-of-sentence)** | Say "Please go to the next slide" | `NEXT_SLIDE` triggers correctly |
| 6 | **Long sentence rejection** | Say a 15-word sentence containing "next slide" | Length penalty fires, `NONE` returned |
| 7 | **Safety cap** | Speak continuously for 30+ seconds without pausing | Force-transcription fires, buffer resets, no OOM |

---

## 8. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Silero VAD model download fails (network) | Module fails to load | `torch.hub.load` caches in `~/.cache/torch/hub/`. First run requires internet. Add clear error message. |
| VAD confidence is unstable for quiet speakers | False negatives (speech not detected) | Threshold is configurable (`VADEngine(threshold=0.5)`). Can lower to `0.35` if needed. Logged per-chunk for tuning. |
| LSTM state corruption across utterances | Confidence bleeds, wrong triggers | `vad.reset()` called after every transcription fire. |
| Length penalty too aggressive (rejects valid long commands) | False negatives for wordy users | Threshold set to 10 words (generous). "Go to slide three" is 4 words. "Can we please skip ahead to the next one" is 9 words. 10 covers all natural command variants. |
| `omegaconf` version conflict with existing packages | pip install failure | Pin `>=2.3.0` which is compatible with PyTorch 2.x and Silero VAD. |
