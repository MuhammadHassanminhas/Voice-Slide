# Phase 4.1b — Optimization Plan: GPU Acceleration & Threshold Tuning

> **Status**: Plan for review — no code changes until approved.
>
> **Objective**: Reduce end-to-end voice-command latency and eliminate missed
> commands by (1) offloading VAD inference to the GPU, (2) reducing the number
> of forward passes per chunk, and (3) tuning confidence/silence thresholds for
> natural, fast speech.

---

## 0. Latency Budget — Where Time Is Spent Today

Before proposing changes, here is the current per-utterance latency breakdown
from the moment the speaker **stops talking** to the moment the frontend
receives a `nav_command`:

| Stage | Current Cost | Bottleneck? |
|---|---|---|
| A. Silence detection (`SILENCE_CHUNKS_THRESHOLD = 2`) | **500 ms** (2 × 250 ms chunks) | **Yes — largest fixed cost** |
| B. VAD inference per chunk (7 × 512-sample windows, CPU) | ~2–3 ms | Moderate — 7 sequential `model()` calls |
| C. `transcribe_chunk` (faster-whisper on GPU) | ~200–500 ms | No — already on CUDA float16 |
| D. `classify_intent` (embedding cosine similarity) | ~10–20 ms | No |
| E. WebSocket round-trip + Reveal.js reaction | ~5 ms | No |
| **Total (end-of-speech → slide change)** | **~720–1030 ms** | |

**Key insight**: Stage A alone accounts for roughly **half** the total latency.
Reducing it from 500 ms to 250 ms is the single biggest win.  Stage B is the
second target — reducing forward passes from 7 to 2 cuts VAD CPU time by ~3.5×
and, combined with GPU offload, frees the eventlet loop to process the next
WebSocket frame sooner.

### Why commands are missed entirely

The VAD confidence threshold (`0.5`) is tuned for studio-clean audio.  In a
typical home/office environment with laptop speakers, gain-controlled mic
input, and natural conversational volume, speech chunks regularly score
`0.35–0.50`.  Those chunks are **classified as silence and discarded**, which
means the speech buffer never accumulates enough data, and the utterance is
silently dropped.

---

## 1. Change 1 — GPU Offload (`vad_engine.py`)

### What Changes

| Current (line) | Change |
|---|---|
| Model loaded on **CPU** (implicit default, line 40) | Auto-detect CUDA; move model to GPU with CPU fallback |
| Tensor created on CPU (line 73) | Transfer full tensor to device **once**, then slice windows on-device |
| No `self._device` attribute | Store resolved device for reuse |

### Detailed Design

```
__init__(threshold=0.35)
│
├── device = "cuda" if torch.cuda.is_available() else "cpu"
├── model, _ = torch.hub.load(...)
├── model = model.to(device)          ← move weights + LSTM state to GPU
├── model.reset_states()
├── self._model = model
├── self._device = device
└── log: "Silero VAD loaded on {device}"
```

```
get_speech_confidence(audio_bytes)
│
├── np.frombuffer(audio_bytes, dtype=np.float32).copy()
├── torch.from_numpy(array).to(self._device)   ← ONE transfer for the full 4000-sample tensor
│
├── for window in 1536-sample slices:           ← slice on-device (zero-copy)
│   └── conf = float(self._model(window, 16000))
└── return max(conf)
```

**Why transfer once, slice on-device**: The current code creates a CPU tensor
and slices 7 windows from it — each `self._model(window, ...)` call would
require an implicit CPU→GPU transfer if the model lives on GPU.  By calling
`.to(self._device)` on the **full** 4000-sample tensor first, all subsequent
`tensor[start:end]` slices are zero-copy views on the same GPU memory.
This reduces GPU transfer operations from N (one per window) to exactly **1**.

### CPU Fallback

Mirrors the pattern in `transcriber.py` (lines 42–47):

```python
try:
    self._device = "cuda"
    self._model = model.to("cuda")
except Exception:
    logger.warning("CUDA unavailable for VAD, falling back to CPU.")
    self._device = "cpu"
```

### Hidden-State Device Safety

Silero VAD's `reset_states()` internally calls `torch.zeros(...)` to
reinitialise the LSTM hidden state.  On some Silero versions, these tensors are
created on CPU regardless of where the model weights live.  After every
`reset()` call, we must ensure the hidden state is on the correct device:

```python
def reset(self) -> None:
    self._model.reset_states()
    # Guard: force hidden state onto the model's device in case
    # reset_states() created them on CPU.
    if self._device != "cpu":
        for attr in ("_h", "_c", "h", "c"):
            t = getattr(self._model, attr, None)
            if t is not None and t.device.type != self._device:
                setattr(self._model, attr, t.to(self._device))
```

This is defensive — it does nothing if the states are already correct, and
silently handles the attribute-name differences between Silero v4 and v5.

---

## 2. Change 2 — Window Size Increase (`vad_engine.py`)

### What Changes

| Current | Proposed |
|---|---|
| `_VAD_WINDOW_SAMPLES = 512` | `_VAD_WINDOW_SAMPLES = 1536` |
| 7 forward passes per 4000-sample chunk | **2** forward passes per chunk |

### Rationale

Silero VAD v5 at 16 kHz accepts window sizes of **512, 1024, or 1536** samples.
Our 250 ms chunk is 4000 samples:

| Window Size | Forward Passes | Samples Covered | Coverage |
|---|---|---|---|
| 512 | 7 | 3584 / 4000 | 89.6 % |
| 1024 | 3 | 3072 / 4000 | 76.8 % |
| **1536** | **2** | **3072 / 4000** | **76.8 %** |

1536 and 1024 cover the same number of samples (the remainder is <1536 and
discarded), but 1536 gives **fewer forward passes** (2 vs 3).  The 23.2 %
uncovered tail (928 samples / 58 ms) is acceptable — if speech is present only
in that tail, the next chunk's first window will catch it.

### Impact

- **CPU**: 2 × ~0.3 ms = **0.6 ms** per chunk (down from ~2.1 ms).
- **GPU**: 2 × ~0.1 ms = **0.2 ms** per chunk, plus a single ~0.02 ms transfer.
- **Net**: ~3.5× fewer model calls, freeing the eventlet loop faster.

---

## 3. Change 3 — Threshold Tuning

### 3a. VAD Confidence Threshold (`vad_engine.py`)

| Current | Proposed |
|---|---|
| `threshold = 0.5` | `threshold = 0.35` |

**Why 0.35?**

- Silero VAD's own documentation recommends **0.5** for "aggressive" filtering
  and **0.3–0.4** for "sensitive" detection.
- Our pipeline has a second layer of filtering (NLP heuristics, intent
  classifier).  A false-positive from VAD (background noise classified as
  speech) is harmless — it just means a silent/noise buffer gets transcribed
  and Whisper returns empty text.  A false-negative (real speech discarded) is
  **fatal** — the command is lost.
- Therefore, we should bias toward **recall** (catch all speech) over
  **precision** (reject all noise).
- 0.35 is the sweet spot: low enough to catch natural/quiet speech, high enough
  to reject dead silence and constant fan noise (which typically scores 0.01–0.10).

### 3b. Silence Chunks Threshold (`app.py`)

| Current | Proposed |
|---|---|
| `SILENCE_CHUNKS_THRESHOLD = 2` (500 ms) | `SILENCE_CHUNKS_THRESHOLD = 1` (250 ms) |

**Why 1 chunk (250 ms)?**

- Voice commands are short imperative phrases.  A natural pause after "next
  slide" is typically 200–400 ms — well within a single 250 ms chunk boundary.
- Reducing to 1 chunk saves **250 ms of fixed latency** on every single
  utterance.
- **Risk of premature firing is low**: for the silence counter to increment, the
  entire 250 ms chunk must score below the VAD threshold.  In natural fast
  speech, inter-word pauses are 50–150 ms — they fall *inside* a chunk that
  also contains speech, so VAD still reports speech.  A full 250 ms of pure
  silence genuinely indicates the speaker has finished.
- **Synergy with the lower VAD threshold (0.35)**: more chunks are correctly
  classified as speech → fewer false-silence frames → even lower risk of
  premature firing.

### Projected Latency After All Changes

| Stage | Before | After | Δ |
|---|---|---|---|
| A. Silence detection | 500 ms | **250 ms** | **−250 ms** |
| B. VAD inference | ~2.5 ms | **~0.2 ms** (GPU) | **−2.3 ms** |
| C. Whisper transcription | ~350 ms | ~350 ms | — |
| D. Intent classification | ~15 ms | ~15 ms | — |
| E. WebSocket + UI | ~5 ms | ~5 ms | — |
| **Total** | **~870 ms** | **~620 ms** | **~−250 ms** |

The dominant win is **250 ms from the silence threshold**.  GPU + window-size
changes contribute ~2 ms of direct speedup but — critically — reduce CPU
contention under the eventlet cooperative loop, which prevents chunk processing
from falling behind during sustained speech.

---

## 4. Files Changed — Summary

| File | Lines Affected | Change |
|---|---|---|
| `backend/vad_engine.py` (line 21) | `_VAD_WINDOW_SAMPLES` | `512` → `1536` |
| `backend/vad_engine.py` (line 29) | `threshold` default | `0.5` → `0.35` |
| `backend/vad_engine.py` (lines 32–48) | `__init__` | Add `self._device`, `.to(device)`, CUDA/CPU fallback |
| `backend/vad_engine.py` (line 73) | `get_speech_confidence` | `.to(self._device)` on full tensor before windowing loop |
| `backend/vad_engine.py` (lines 96–98) | `reset` | Add hidden-state device guard after `reset_states()` |
| `backend/app.py` (line 158) | `SILENCE_CHUNKS_THRESHOLD` | `2` → `1` |

**No other files are touched.**  The public API (`get_vad_engine()`,
`get_speech_confidence()`, `is_speech()`, `reset()`) remains identical.
`app.py` only changes one constant.

---

## 5. Execution Order

1. **`backend/vad_engine.py`** — Apply all four changes (window size, threshold,
   GPU offload, reset guard).
2. **`backend/app.py`** — Change `SILENCE_CHUNKS_THRESHOLD` from `2` to `1`.
3. **Smoke test** — Run the same verification from Phase 4.1:
   ```
   python -c "from vad_engine import get_vad_engine; e = get_vad_engine(); ..."
   ```
   Verify log says `"Silero VAD loaded on cuda"`.
4. **Silence regression test** — Feed a zero-array, confirm confidence < 0.35.
5. **Live test** — Start server, speak naturally, verify faster response and no
   missed commands.

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Silero JIT model doesn't support `.to('cuda')` | Low | Falls back to CPU (no regression) | try/except with CPU fallback |
| `reset_states()` creates hidden state on wrong device | Medium | RuntimeError on next inference | Defensive device guard in `reset()` |
| Threshold 0.35 too sensitive (fan noise triggers speech) | Low | Noise gets transcribed, Whisper returns "", no nav command | Harmless — Whisper + intent classifier filter it out |
| `SILENCE_CHUNKS_THRESHOLD = 1` fires mid-phrase | Very Low | Phrase split into two transcriptions | Inter-word pauses (50–150 ms) don't fill a 250 ms chunk; lower VAD threshold further reduces false-silence risk |
| Larger windows (1536) miss speech in the uncovered 928-sample tail | Very Low | Detected by next chunk's first window with ~60 ms delay | Acceptable — 60 ms is imperceptible |
