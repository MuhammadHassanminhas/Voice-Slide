# Phase 7 — Live Transcription & Subtitles with Translation

## Problem Statement

The current transcript bar (Phase 3) is a simple single-line `<p>` element that gets overwritten on every utterance. It lacks:

1. **History** — previous lines vanish instantly.
2. **TV-style presentation** — no smooth scrolling, no multi-line rolling display.
3. **Translation** — English-only; no way for a multilingual audience to follow along.

**Solution:** Upgrade the transcript bar into a professional TV-style subtitle overlay with rolling 2-line display and smooth scrolling, powered by lazy-loaded MarianMT models for real-time translation into 5 target languages.

---

## 1. Dependencies & Models

### 1.1 Python Packages

| Package | Status | Purpose |
|---------|--------|---------|
| `transformers>=4.40.0` | ✅ Already installed (5.2.0) | MarianMT model loading |
| `sentencepiece>=0.2.0` | ❌ **Needs install** | Tokenizer backend for MarianMT |

Add to `requirements.txt`:
```
# Phase 7 Dependencies — Live Translation (MarianMT)
sentencepiece>=0.2.0
# NOTE: transformers is already installed via sentence-transformers dependency
```

### 1.2 Helsinki-NLP MarianMT Model Paths

All models follow the `Helsinki-NLP/opus-mt-{src}-{tgt}` naming convention. These are auto-downloaded by `transformers` on first use (~300 MB each).

| Language | Model ID | Size | Notes |
|----------|----------|------|-------|
| Spanish | `Helsinki-NLP/opus-mt-en-es` | ~300 MB | Well-tested, high quality |
| French | `Helsinki-NLP/opus-mt-en-fr` | ~300 MB | Well-tested, high quality |
| German | `Helsinki-NLP/opus-mt-en-de` | ~300 MB | Well-tested, high quality |
| Urdu | `Helsinki-NLP/opus-mt-en-ur` | ~300 MB | Smaller training corpus; adequate |
| Chinese | `Helsinki-NLP/opus-mt-en-zh` | ~300 MB | Simplified Chinese output |

**Language code mapping** (used in API and WebSocket payloads):
```python
SUPPORTED_LANGUAGES = {
    "en": None,             # English — no translation (passthrough)
    "es": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ur": "Helsinki-NLP/opus-mt-en-ur",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
}
```

### 1.3 RAM Budget Analysis

| State | Models in RAM | Estimated RAM |
|-------|--------------|---------------|
| Presentation (no translation) | faster-whisper base + all-MiniLM-L6-v2 + Silero VAD + thefuzz | ~1.2 GB |
| Presentation + 1 translation model | Above + 1 MarianMT model | ~1.8 GB |
| Max during live session | Above (never load >1 MarianMT) | ~1.8 GB |

**Within the 4 GB budget.** Only one MarianMT model is loaded at a time. Switching languages unloads the previous model first.

---

## 2. Backend Architecture

### 2.1 New Module: `backend/translator.py`

```
Responsibilities:
- Lazy-load MarianMT model + tokenizer on first translation request
- Unload current model when language changes (memory safety)
- Expose a simple translate(text, target_lang) function
- Thread-safe singleton pattern (matches transcriber.py / embeddings.py)
```

#### 2.1.1 Module State

```python
_current_lang: str = "en"        # Currently loaded language code
_model = None                     # MarianMTModel instance or None
_tokenizer = None                 # MarianTokenizer instance or None
```

#### 2.1.2 Lazy Loading Strategy

```python
def _load_model(lang_code: str) -> None:
    """Load the MarianMT model for the given language. Unloads any previous model first."""
    global _model, _tokenizer, _current_lang

    if lang_code == "en" or lang_code not in SUPPORTED_LANGUAGES:
        # English = passthrough, no model needed
        _unload_model()
        _current_lang = lang_code
        return

    if _current_lang == lang_code and _model is not None:
        return  # Already loaded

    _unload_model()  # Free RAM from previous model

    model_id = SUPPORTED_LANGUAGES[lang_code]
    logger.info("Loading translation model: %s", model_id)
    _tokenizer = MarianTokenizer.from_pretrained(model_id)
    _model = MarianMTModel.from_pretrained(model_id)

    # Move to GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    _model.eval()

    _current_lang = lang_code
    logger.info("Translation model loaded: %s → %s", model_id, device)


def _unload_model() -> None:
    """Free the current translation model from memory."""
    global _model, _tokenizer
    if _model is not None:
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        torch.cuda.empty_cache()  # Reclaim GPU memory
        logger.info("Previous translation model unloaded.")
```

**Key design point:** `_load_model` is called lazily — only when the first translation request arrives for a given language. If the user never enables translation, zero extra RAM is consumed.

#### 2.1.3 Public API

```python
def set_language(lang_code: str) -> None:
    """Set the target translation language. 'en' disables translation."""
    _load_model(lang_code)


def translate(text: str) -> str:
    """Translate English text to the current target language.

    Returns the original text unchanged if lang is 'en' or no model is loaded.
    """
    if _current_lang == "en" or _model is None:
        return text

    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        translated = _model.generate(**inputs)

    return _tokenizer.decode(translated[0], skip_special_tokens=True)


def get_current_language() -> str:
    """Return the currently active language code."""
    return _current_lang
```

### 2.2 Changes to `app.py`

#### 2.2.1 Import

```python
import translator
```

Added alongside the other Phase 4–6 imports (around line 180).

#### 2.2.2 Translation in the Transcript Pipeline

The transcript emit at line 263 currently sends the raw English text:

```python
socketio.emit("transcript", result)
```

This becomes a **dual emit** — both the original English transcript (for intent classification, which must stay English) and the translated version (for display):

```python
# Emit original English transcript for subtitle display
original_text = transcribed_text

# Translate if a non-English language is active
translated_text = translator.translate(original_text)

socketio.emit("transcript", {
    "text": original_text,
    "translated_text": translated_text,
    "language": translator.get_current_language(),
    "is_final": result.get("is_final", True),
})
```

**Critical:** Intent classification, context search, and the Phase 6 interceptor all continue to use `transcribed_text` (English). Only the frontend display uses `translated_text`.

#### 2.2.3 New WebSocket Event: `set_language`

```python
@socketio.on("set_language")
def handle_set_language(data):
    lang_code = data.get("language", "en") if isinstance(data, dict) else "en"
    translator.set_language(lang_code)
    logger.info("Translation language set → %s", lang_code)
    socketio.emit("language_changed", {"language": lang_code})
```

The `language_changed` emit confirms the switch to the frontend (useful for UI state sync and showing a toast).

---

## 3. Frontend UI & WebSockets

### 3.1 HTML Changes (`index.html`)

Add a subtitle controls bar between the mic button and the transcript bar:

```html
<!-- ── Subtitle Controls (Phase 7) ─────────────────────────────────── -->
<div id="subtitle-controls" class="subtitle-controls">
  <label class="subtitle-toggle">
    <input type="checkbox" id="subtitle-toggle-checkbox" checked />
    <span class="subtitle-toggle__slider"></span>
    <span class="subtitle-toggle__label">Subtitles</span>
  </label>

  <div class="language-selector">
    <select id="language-select" class="language-select">
      <option value="en" selected>🇬🇧 English</option>
      <option value="es">🇪🇸 Spanish</option>
      <option value="fr">🇫🇷 French</option>
      <option value="de">🇩🇪 German</option>
      <option value="ur">🇵🇰 Urdu</option>
      <option value="zh">🇨🇳 Chinese</option>
    </select>
  </div>
</div>
```

**Placement:** Fixed position, bottom-left, above the mic button. Only visible when the mic is recording.

### 3.2 Upgraded Transcript Bar HTML

Replace the current simple transcript bar with a rolling 2-line subtitle overlay:

```html
<div id="transcript-bar" class="transcript-bar">
  <div class="subtitle-overlay">
    <div id="subtitle-lines" class="subtitle-lines">
      <!-- Lines are added dynamically, max 2 visible -->
    </div>
  </div>
</div>
```

### 3.3 CSS: Glassmorphism Subtitle Overlay (`presentation.css`)

The current `.transcript-bar` and related styles (lines 388–451) will be **replaced**:

```css
/* ── Phase 7: TV-Style Subtitle Overlay ────────────────────────────────── */

.transcript-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 0 40px 24px;
    transform: translateY(100%);
    transition: transform var(--transition-normal);
    z-index: 999;
    pointer-events: none;
    display: flex;
    justify-content: center;
}

.transcript-bar.visible {
    transform: translateY(0);
}

.subtitle-overlay {
    background: rgba(10, 14, 23, 0.80);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 16px 28px;
    max-width: 860px;
    width: 100%;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    overflow: hidden;
}

.subtitle-lines {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 4.6em;            /* Exactly 2 lines visible */
    overflow: hidden;
}

.subtitle-line {
    font-size: 1.3rem;
    font-family: var(--font-body);
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.7;
    margin: 0;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.6);
    animation: subtitle-fade-in 0.3s ease forwards;
    opacity: 0;
}

.subtitle-line--translated {
    color: var(--accent-primary);
    font-size: 1.15rem;
    font-style: italic;
}

.subtitle-line--fading {
    animation: subtitle-fade-out 0.3s ease forwards;
}

@keyframes subtitle-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes subtitle-fade-out {
    from { opacity: 1; transform: translateY(0); }
    to   { opacity: 0; transform: translateY(-8px); }
}
```

**Design approach:**
- `.subtitle-overlay` uses glassmorphism: semi-transparent dark background + `backdrop-filter: blur(20px)` + subtle border glow.
- `.subtitle-lines` has `max-height: 4.6em` (2 lines × `line-height: 1.7` + gap) and `overflow: hidden`. Older lines are pushed up and removed, achieving the "rolling" effect.
- New lines animate in from below with `subtitle-fade-in`.
- Lines being removed animate out with `subtitle-fade-out`.

### 3.4 Subtitle Controls CSS

```css
/* ── Phase 7: Subtitle Controls ────────────────────────────────────────── */

.subtitle-controls {
    position: fixed;
    bottom: 92px;                   /* Above mic button (56px + 24px + gap) */
    left: 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    z-index: 1001;
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--transition-normal);
}

.subtitle-controls.visible {
    opacity: 1;
    pointer-events: auto;
}

.subtitle-toggle {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    user-select: none;
}

.subtitle-toggle__slider {
    width: 40px;
    height: 22px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 11px;
    position: relative;
    transition: background var(--transition-fast);
}

.subtitle-toggle__slider::after {
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--text-secondary);
    transition: transform var(--transition-fast), background var(--transition-fast);
}

.subtitle-toggle input:checked + .subtitle-toggle__slider {
    background: rgba(0, 200, 255, 0.3);
}

.subtitle-toggle input:checked + .subtitle-toggle__slider::after {
    transform: translateX(18px);
    background: var(--accent-primary);
}

.subtitle-toggle input {
    display: none;
}

.subtitle-toggle__label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: var(--font-body);
}

.language-select {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 0.8rem;
    padding: 6px 10px;
    cursor: pointer;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.language-select:focus {
    outline: none;
    border-color: var(--accent-primary);
}
```

### 3.5 JavaScript State Management (`app.js`)

#### 3.5.1 New DOM References

```javascript
const subtitleControls = document.getElementById("subtitle-controls");
const subtitleToggle = document.getElementById("subtitle-toggle-checkbox");
const languageSelect = document.getElementById("language-select");
const subtitleLines = document.getElementById("subtitle-lines");
```

#### 3.5.2 Subtitle State

```javascript
let subtitlesEnabled = true;
const MAX_SUBTITLE_LINES = 2;
```

#### 3.5.3 Language Change Handler

```javascript
languageSelect.addEventListener("change", (e) => {
    const lang = e.target.value;
    socket.emit("set_language", { language: lang });
});

socket.on("language_changed", (data) => {
    if (data && data.language) {
        const langName = languageSelect.querySelector(
            `option[value="${data.language}"]`
        )?.textContent || data.language;
        showToast(`🌐 Subtitles: ${langName}`);
    }
});
```

#### 3.5.4 Subtitle Toggle Handler

```javascript
subtitleToggle.addEventListener("change", (e) => {
    subtitlesEnabled = e.target.checked;
    transcriptBar.classList.toggle("visible", subtitlesEnabled && isRecording);
});
```

Also update `startRecording()` and `stopRecording()` to show/hide subtitle controls:

```javascript
// In startRecording(), after isRecording = true:
subtitleControls.classList.add("visible");
if (subtitlesEnabled) transcriptBar.classList.add("visible");

// In stopRecording(), after isRecording = false:
subtitleControls.classList.remove("visible");
transcriptBar.classList.remove("visible");
```

#### 3.5.5 Updated Transcript Handler (Rolling 2-Line Display)

Replace the current `socket.on("transcript")` handler:

```javascript
socket.on("transcript", (data) => {
    if (!data || !data.text || !subtitlesEnabled) return;

    const displayText = (data.language && data.language !== "en" && data.translated_text)
        ? data.translated_text
        : data.text;

    addSubtitleLine(displayText, data.language !== "en");
});

function addSubtitleLine(text, isTranslated) {
    const line = document.createElement("p");
    line.className = "subtitle-line";
    if (isTranslated) {
        line.classList.add("subtitle-line--translated");
    }
    line.textContent = text;
    subtitleLines.appendChild(line);

    // Remove excess lines (keep last MAX_SUBTITLE_LINES)
    while (subtitleLines.children.length > MAX_SUBTITLE_LINES) {
        const oldest = subtitleLines.firstElementChild;
        oldest.classList.add("subtitle-line--fading");
        oldest.addEventListener("animationend", () => oldest.remove());
    }
}
```

**How the rolling effect works:**
1. Each new transcript arrives → creates a `<p class="subtitle-line">` → appends to `#subtitle-lines`.
2. The container has `max-height: 4.6em` and `overflow: hidden` — only the last 2 lines are visible.
3. When a third line is added, the oldest gets a `subtitle-line--fading` class (fade-out + slide-up animation), then is removed from the DOM on `animationend`.
4. New lines use `subtitle-fade-in` animation (fade-in + slide-up from below).

This creates the smooth TV-style rolling subtitle effect.

---

## 4. Pipeline Flow Diagram

```
                        ┌─────────────────────┐
                        │   Audio Chunk (PCM)  │
                        └──────────┬──────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   faster-whisper     │
                        │   transcribe_chunk() │
                        └──────────┬──────────┘
                                   ▼
                        transcribed_text (English)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
            classify_intent()   translator.     fuzzy_match /
            (always English)    translate()     context_search
                    │              │            (always English)
                    ▼              ▼
            nav_command     socketio.emit("transcript", {
                              text: original,
                              translated_text: translated,
                              language: lang
                            })
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Frontend   │
                            │  subtitle   │
                            │  overlay    │
                            └─────────────┘
```

**Critical invariant:** Intent classification, context search, and the Phase 6 interceptor always operate on the **original English text**. Translation is a display-only concern applied **after** all NLP processing.

---

## 5. Testing Strategy

### 5.1 New Test File: `tests/test_translator.py`

#### Unit Tests for `translator.py`

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `test_default_language_is_english` | `get_current_language()` returns `"en"` on startup |
| 2 | `test_set_language_english_passthrough` | `set_language("en")` → no model loaded, `_model` stays None |
| 3 | `test_translate_english_returns_original` | `translate("hello")` with lang `"en"` returns `"hello"` unchanged |
| 4 | `test_set_language_unsupported_stays_english` | `set_language("xx")` → stays on `"en"`, no crash |
| 5 | `test_set_language_loads_model` | `set_language("es")` → `_model` is not None, `_current_lang == "es"` |
| 6 | `test_translate_produces_output` | `set_language("es")` → `translate("Hello world")` returns non-empty Spanish text |
| 7 | `test_switch_language_unloads_previous` | Load "es" → load "fr" → verify previous model was unloaded (only 1 model in memory) |
| 8 | `test_translate_empty_string` | `translate("")` returns `""` without error |
| 9 | `test_get_current_language_after_switch` | `set_language("de")` → `get_current_language()` returns `"de"` |
| 10 | `test_set_language_same_no_reload` | `set_language("es")` twice → model is not re-downloaded/re-loaded |

> **Note:** Tests 5, 6, 7, and 10 require actual model downloads (~300 MB). These should be marked with `@pytest.mark.slow` so they can be skipped in fast CI runs. For the core test run, tests 1–4 and 8–9 can use mocks.

### 5.2 New Integration Tests: `tests/test_transcript_pipeline.py`

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `test_transcript_emit_includes_translated_text` | Mock translator → verify `socketio.emit("transcript")` payload contains `translated_text` key |
| 2 | `test_transcript_english_no_translation` | Language = "en" → `translated_text` equals `text` |
| 3 | `test_set_language_event_calls_translator` | Emit `set_language` → verify `translator.set_language()` is called |
| 4 | `test_language_changed_emitted` | After `set_language` → verify `language_changed` event is emitted back |
| 5 | `test_intent_classification_uses_english` | Language = "es" → verify `classify_intent()` receives original English text, not translated |

### 5.3 Existing Test Preservation

All 99 existing tests **must continue to pass** unchanged. The transcript emit payload is expanded (new fields added) but the `text` key remains, so existing mocks that check `result.get("text")` are unaffected.

**Expected final test count: 114** (99 existing + 10 translator + 5 pipeline)

---

## 6. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/translator.py` | **CREATE** | MarianMT lazy-loading, translate(), set_language(), unload |
| `backend/app.py` | **EDIT** | Import translator, dual emit (text + translated_text), add `set_language` handler |
| `backend/requirements.txt` | **EDIT** | Add `sentencepiece>=0.2.0` |
| `frontend/index.html` | **EDIT** | Add subtitle controls (toggle + language dropdown), replace transcript bar HTML |
| `frontend/js/app.js` | **EDIT** | New DOM refs, language change handler, subtitle toggle, rolling 2-line display, updated transcript handler |
| `frontend/css/presentation.css` | **EDIT** | Replace transcript bar styles with glassmorphism subtitle overlay + controls |
| `tests/test_translator.py` | **CREATE** | 10 unit tests for translator.py |
| `tests/test_transcript_pipeline.py` | **CREATE** | 5 integration tests for the translation pipeline |

---

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| MarianMT model download is slow (~300 MB per language) | First-time download only; cached in `~/.cache/huggingface/`. Show "Loading translation model…" toast while downloading. |
| Translation latency blocks the pipeline | `translator.translate()` runs on GPU if available (<50ms per sentence). Utterances are short (1–3 sentences). If latency is a concern, we can move translation to a background thread in a future phase. |
| GPU OOM with MarianMT + faster-whisper + sentence-transformers | MarianMT models are small (~300 MB). Combined with existing models, peak GPU usage is ~2 GB — well within typical 4–8 GB cards. `_unload_model()` frees GPU memory via `torch.cuda.empty_cache()`. |
| `sentencepiece` C extension fails to compile | Pre-built wheels exist for Windows/Linux/macOS on Python 3.10–3.12. Extremely unlikely to fail. |
| Existing tests break from changed transcript payload | The `text` key in the transcript emit is unchanged. New fields (`translated_text`, `language`) are additive. Existing mocks check `result.get("text")` which continues to work. |
| Translation quality for Urdu/Chinese | Helsinki-NLP opus-mt models are adequate for subtitle-quality translation. Not production translation, but sufficient for a live presentation aid. |

---

*Awaiting approval before generating code.*
