# Phase 8 — Smart Q&A Assistant & Presenter Panel

## Problem Statement

During a live presentation, audience members ask questions that relate to specific slides or notes the presenter has prepared. Currently:

1. **No question detection** — the system cannot distinguish questions from statements or navigation commands.
2. **No notes surfacing** — slide `notes` fields exist in `slides.json` but are never presented to the speaker in real-time.
3. **No private presenter view** — the speaker has no dashboard to see suggested talking points or answers.

**Solution:** Build a Q&A assistant module that detects spoken questions from the live transcript, searches slide notes using the existing `all-MiniLM-L6-v2` embedding model (zero additional RAM), and pushes the top 3 relevant notes to a private `/presenter` dashboard via WebSockets.

---

## 1. State & Memory Management

### 1.1 Accessing Slide Notes

Slide notes are already loaded into memory by `context_search.build_index()` at startup (line 55 of `app.py`). The `_extract_body()` function in `context_search.py` (line 66) concatenates notes into the body text alongside bullets, captions, and quotes — but notes are **mixed in** with other content, making them unsuitable for isolated retrieval.

**Approach:** `qa_assistant.py` will build its own **notes-only index** by reading the raw slide dicts from `context_search._slides` (already stored in memory at `context_search.py` line 26). This avoids re-loading `slides.json` from disk.

```python
# qa_assistant.py reads notes from the already-loaded slide data:
import context_search

def _extract_notes() -> list[dict]:
    """Extract notes from the slides already loaded in context_search._slides."""
    entries = []
    for idx, slide in enumerate(context_search._slides):
        note = slide.get("notes", "").strip()
        if note:
            heading = slide.get("heading", f"Slide {idx + 1}")
            entries.append({
                "slide_index": idx,
                "heading": heading,
                "note_text": note,
            })
    return entries
```

No file I/O. No duplicate data. `context_search._slides` is populated by `build_index()` which already runs at app startup and after every upload/save.

### 1.2 Reusing the Embedding Model (Critical — Zero Extra RAM)

The `all-MiniLM-L6-v2` model is managed as a singleton in `backend/embeddings.py`:

```
embeddings.py
├── _model_instance = None          (module-level singleton)
├── get_embedding_model()           (lazy-loads once, returns cached instance)
└── encode(texts: List[str])        (public API — calls get_embedding_model() internally)
```

Both `intent_classifier.py` and `context_search.py` already import and call `embeddings.encode()`. The model is loaded exactly once into RAM (~90 MB) and shared.

**`qa_assistant.py` will follow the identical pattern:**

```python
from embeddings import encode          # Same import as intent_classifier.py
from sentence_transformers import util  # Same import as context_search.py
```

This guarantees:
- **Zero additional model loading** — `encode()` calls `get_embedding_model()` which returns the existing `_model_instance`.
- **Same device placement** — if the model is on CUDA, note embeddings also compute on CUDA.
- **Consistent embedding space** — questions and notes are encoded by the same model, ensuring valid cosine similarity.

### 1.3 RAM Budget Impact

| State | Models in RAM | Estimated RAM |
|-------|--------------|---------------|
| Phase 6 (current) | faster-whisper base + all-MiniLM-L6-v2 + Silero VAD | ~1.2 GB |
| Phase 8 (Q&A active) | Same — no new models, only a small tensor of note embeddings | ~1.2 GB + ~50 KB |

**Impact: Negligible.** The only new memory is a tensor of shape `(N, 384)` where N = number of slides with non-empty notes. Even with 100 slides, that's `100 × 384 × 4 bytes = ~150 KB`.

---

## 2. Question Detection & Retrieval Logic (`qa_assistant.py`)

### 2.1 Module Responsibilities

```
qa_assistant.py
├── _extract_notes()           — Pull notes from context_search._slides
├── build_notes_index()        — Embed all notes (called once at startup / re-index)
├── is_question(text)          — Heuristic question detector
├── search_notes(question)     — Cosine similarity → top 3 results
└── Module state:
    ├── _note_entries: list[dict]      — [{slide_index, heading, note_text}, ...]
    └── _note_embeddings: Tensor|None  — (N × 384) or None
```

### 2.2 Question Detection Heuristics

Speech-to-text (faster-whisper) does **not** reliably produce `?` punctuation. We cannot depend on it. Instead, we use a regex-based approach that checks for interrogative patterns at the **start** of the utterance.

```python
import re

_QUESTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:what|which|where|when|who|whom|whose|why|how)"             # WH-words
    r"|(?:is|are|was|were|do|does|did|can|could|will|would|shall"
    r"|should|may|might|has|have|had)"                              # Aux-initial
    r")\b",
    re.IGNORECASE,
)

# Secondary: catch trailing "?" if whisper does produce it
_TRAILING_QUESTION_RE = re.compile(r"\?\s*$")

def is_question(text: str) -> bool:
    """Detect if a transcript segment is a question.

    Uses a two-tier heuristic:
    1. Check if the utterance starts with a WH-word or auxiliary verb
       (interrogative word order).
    2. Check for a trailing '?' (unreliable from STT but free to check).

    Short utterances (< 3 words) are excluded to avoid false positives
    from filler words like "what" or "how".
    """
    stripped = text.strip()
    words = stripped.split()
    if len(words) < 3:
        return False

    if _TRAILING_QUESTION_RE.search(stripped):
        return True

    return bool(_QUESTION_RE.match(stripped))
```

**Design rationale:**

| Pattern | Example | Detected? |
|---------|---------|-----------|
| WH-word start | "What is the revenue for Q3" | ✅ Yes — `what` at start |
| Aux-verb start | "Can you explain the architecture" | ✅ Yes — `can` at start |
| Trailing `?` | "the budget is how much?" | ✅ Yes — trailing `?` |
| Statement | "Next slide please" | ❌ No — no interrogative pattern |
| Slide content | "Revenue grew 18 percent" | ❌ No — declarative |
| Short filler | "What" / "How" | ❌ No — fewer than 3 words |
| Emphasis | "This is really important" | ❌ No — no interrogative pattern |

**Why not use the embedding model for question detection?** Embedding similarity would require a training set of question vs. non-question utterances. The regex heuristic is deterministic, zero-latency, and handles the specific patterns faster-whisper produces. Semantic matching is reserved for the **retrieval** step (matching the detected question against notes).

### 2.3 Notes Index Construction

```python
_note_entries: list[dict] = []
_note_embeddings = None  # Tensor or None


def build_notes_index() -> None:
    """Embed all non-empty slide notes for semantic search.

    Reads from context_search._slides (already in memory).
    Safe to call multiple times — each call replaces the previous index.
    """
    global _note_entries, _note_embeddings

    _note_entries = _extract_notes()

    if _note_entries:
        texts = [e["note_text"] for e in _note_entries]
        _note_embeddings = encode(texts)
        logger.info("Q&A notes index built: %d notes embedded.", len(_note_entries))
    else:
        _note_embeddings = None
        logger.info("Q&A notes index: no notes found in slides.")
```

**When is it called?**
- At app startup, immediately after `_index_current_slides()` in `app.py`.
- After every `api_upload_pptx()` and `api_save_slides()` — same places that already call `context_search.build_index()`.

### 2.4 Retrieval: Top 3 Note Search

```python
NOTES_SEARCH_THRESHOLD = 0.25  # Lower than context_search (0.30) — notes are short

def search_notes(question: str, top_k: int = 3) -> list[dict]:
    """Find the top-k slide notes most relevant to the question.

    Returns a list of dicts:
        [{"slide_index": int, "heading": str, "note_text": str, "score": float}, ...]

    Returns an empty list if no notes are indexed or no match exceeds threshold.
    """
    if _note_embeddings is None or not _note_entries:
        return []

    query_embedding = encode([question])
    hits = util.semantic_search(query_embedding, _note_embeddings, top_k=top_k)

    results = []
    for hit in hits[0]:
        if hit["score"] >= NOTES_SEARCH_THRESHOLD:
            entry = _note_entries[hit["corpus_id"]]
            results.append({
                "slide_index": entry["slide_index"],
                "heading": entry["heading"],
                "note_text": entry["note_text"],
                "score": round(float(hit["score"]), 3),
            })

    return results
```

**Why `top_k=3`?** A presenter dashboard should show a small, scannable set of suggestions. Three cards give enough context without overwhelming. The threshold (0.25) is deliberately lower than `context_search.SEARCH_THRESHOLD` (0.30) because slide notes are often short, terse phrases that may only partially match a spoken question.

---

## 3. Backend Routing & WebSockets (`app.py`)

### 3.1 New Import

```python
import qa_assistant
```

Added alongside the existing Phase 4–6 imports (around line 181).

### 3.2 New Flask Route: `/presenter`

```python
@app.route("/presenter")
def presenter_page():
    """Serve the private Presenter Panel page."""
    return send_from_directory(config.FRONTEND_DIR, "presenter.html")
```

Placed alongside the existing `index()` and `upload_page()` routes (around line 74). Follows the identical pattern: a simple `send_from_directory` call.

**Access model:** The presenter opens `/presenter` in a separate browser tab/window (or on a second screen). The same Flask server serves both pages. Both pages connect to the same Socket.IO server and receive events independently.

### 3.3 Index Rebuild Hooks

The notes index must rebuild whenever slides change. Three locations in `app.py` already call `context_search.build_index()` — each gets a companion call:

```python
# 1. Startup (_index_current_slides, ~line 54):
context_search.build_index(data.get("slides", []))
qa_assistant.build_notes_index()  # ← NEW

# 2. After PPTX upload (api_upload_pptx, ~line 140):
context_search.build_index(slide_data.get("slides", []))
qa_assistant.build_notes_index()  # ← NEW

# 3. After JSON save (api_save_slides, ~line 165):
context_search.build_index(data.get("slides", []))
qa_assistant.build_notes_index()  # ← NEW
```

**Why not call `build_notes_index()` inside `context_search.build_index()`?** Separation of concerns. `context_search` owns content indexing for navigation. `qa_assistant` owns notes indexing for Q&A. They share the same embedding model but have independent indices.

### 3.4 Question Interception in `_process_speech_buffer()`

The question check is inserted **after** the transcript emit (line 262) and **before** intent classification (line 264). This ensures:
- The transcript is always displayed to the audience (no delay).
- Question detection runs in parallel with intent classification.
- A question can **also** be a navigation command (e.g., "Can you go to the budget slide?") — intent classification handles that. The Q&A result is additive, not blocking.

```python
def _process_speech_buffer():
    # ... (existing: snapshot buffer, transcribe, emit transcript) ...

    socketio.emit("transcript", result)                  # line 262 (existing)

    # ── Phase 8: Q&A — detect questions, search notes ──────────────
    if qa_assistant.is_question(transcribed_text):
        qa_results = qa_assistant.search_notes(transcribed_text)
        if qa_results:
            socketio.emit("qa_update", {
                "question": transcribed_text,
                "results": qa_results,
            })
            logger.info(
                "Q&A: '%s' → %d note(s) found",
                transcribed_text[:60], len(qa_results),
            )

    # 2. Classify intent                                  # line 264 (existing)
    intent_result = classify_intent(transcribed_text)
    # ... (rest of pipeline unchanged) ...
```

**Critical design decisions:**
1. **Non-blocking:** The `qa_update` emit goes to `/presenter` clients. The main presentation page (`index.html`) ignores this event. The pipeline continues to intent classification regardless.
2. **No cooldown:** Unlike `nav_command`, questions can fire rapidly without a cooldown. The presenter panel simply updates with the latest results.
3. **Additive only:** If the same utterance is both a question ("Can you go to the budget slide?") and a navigation command, **both** fire: `qa_update` to the presenter panel AND `nav_command` to the presentation. Neither blocks the other.

### 3.5 Pipeline Flow with Phase 8

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
                    ┌──────────────┼────────────────┐
                    ▼              ▼                 ▼
          socketio.emit      is_question()     classify_intent()
          ("transcript")          │                  │
                           ┌─────┴─────┐      ┌─────┴─────┐
                           │  Yes      │  No  │ Command?   │
                           ▼           │      ▼            ▼
                     search_notes()    │  nav_command   interceptor/
                           │           │                fallback
                           ▼           │
                   socketio.emit       │
                   ("qa_update")       │
                        │              │
                        ▼              │
                ┌──────────────┐       │
                │  /presenter  │       │
                │   dashboard  │       │
                └──────────────┘       │
                                       └──── (discarded)
```

---

## 4. Frontend UI

### 4.1 New Files

| File | Purpose |
|------|---------|
| `frontend/presenter.html` | Presenter Panel page structure |
| `frontend/js/presenter.js` | WebSocket listener + dynamic card rendering |
| `frontend/css/presenter.css` | Dashboard layout + card styles |

### 4.2 Presenter Panel Layout (`presenter.html`)

```
┌─────────────────────────────────────────────────────────────────────┐
│  VoiceSlide                         [Back to Presentation]          │
│  ─ Presenter Panel ─                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ Status Bar ──────────────────────────────────────────────────┐  │
│  │  🟢 Connected  •  Listening for questions...                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Last Question ───────────────────────────────────────────────┐  │
│  │  "What is the revenue for Q3?"                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Suggested Notes ─────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌─ Card 1 ─────────────────────────────────────────────┐     │  │
│  │  │  📌 Slide 4: Revenue Chart              Score: 0.82  │     │  │
│  │  │  ─────────────────────────────────────────────────── │     │  │
│  │  │  "Q3 revenue was 680 million dollars."               │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                               │  │
│  │  ┌─ Card 2 ─────────────────────────────────────────────┐     │  │
│  │  │  📌 Slide 8: Financial Overview         Score: 0.64  │     │  │
│  │  │  ─────────────────────────────────────────────────── │     │  │
│  │  │  "Full-year projection is 2.7 billion."              │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                               │  │
│  │  ┌─ Card 3 ─────────────────────────────────────────────┐     │  │
│  │  │  📌 Slide 12: Quarterly Breakdown       Score: 0.51  │     │  │
│  │  │  ─────────────────────────────────────────────────── │     │  │
│  │  │  "Q3 was the strongest quarter due to holiday..."    │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ History ─────────────────────────────────────────────────────┐  │
│  │  Previous questions appear here (newest first, max 10)        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key UI elements:**

| Element | ID | Purpose |
|---------|----|---------|
| Status bar | `#connection-status` | Shows WebSocket connection state (🟢 Connected / 🔴 Disconnected) |
| Last question | `#last-question` | Displays the most recently detected question text |
| Note cards container | `#qa-cards` | Holds the 1–3 dynamically rendered note cards |
| History list | `#qa-history` | Scrollable list of previous Q&A pairs (max 10, newest first) |

### 4.3 HTML Structure (`presenter.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VoiceSlide — Presenter Panel</title>
  <!-- Same Google Fonts as index.html -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/css/style.css" />
  <link rel="stylesheet" href="/css/presenter.css" />
</head>
<body>
  <header class="topbar">
    <div class="topbar__logo">VoiceSlide</div>
    <span class="topbar__subtitle">Presenter Panel</span>
    <nav class="topbar__nav">
      <a href="/" class="btn-text">Back to Presentation</a>
    </nav>
  </header>

  <main class="presenter-container">
    <!-- Status -->
    <div id="connection-status" class="status-bar">
      <span class="status-dot"></span>
      <span id="status-text">Connecting...</span>
    </div>

    <!-- Last Question -->
    <section class="qa-section">
      <h2 class="qa-section__title">Last Question</h2>
      <div id="last-question" class="last-question">
        <p class="last-question__text">No questions detected yet.</p>
      </div>
    </section>

    <!-- Suggested Notes -->
    <section class="qa-section">
      <h2 class="qa-section__title">Suggested Notes</h2>
      <div id="qa-cards" class="qa-cards">
        <p class="qa-cards__empty">Waiting for a question...</p>
      </div>
    </section>

    <!-- History -->
    <section class="qa-section">
      <h2 class="qa-section__title">History</h2>
      <div id="qa-history" class="qa-history">
        <p class="qa-history__empty">No history yet.</p>
      </div>
    </section>
  </main>

  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <script src="/js/presenter.js"></script>
</body>
</html>
```

Follows the same structure as `upload.html`: topbar header with logo and nav link, a `<main>` container, shared `style.css` for design tokens, and a page-specific CSS file.

### 4.4 JavaScript: WebSocket Listener (`presenter.js`)

```javascript
"use strict";

const statusDot      = document.querySelector(".status-dot");
const statusText     = document.getElementById("status-text");
const lastQuestionEl = document.getElementById("last-question");
const qaCardsEl      = document.getElementById("qa-cards");
const qaHistoryEl    = document.getElementById("qa-history");

const MAX_HISTORY = 10;
const socket = io();

// ── Connection Status ──────────────────────────────────────────────────
socket.on("connect", () => {
    statusDot.classList.add("connected");
    statusText.textContent = "Connected — Listening for questions...";
});

socket.on("disconnect", () => {
    statusDot.classList.remove("connected");
    statusText.textContent = "Disconnected";
});

// ── Q&A Update Handler ─────────────────────────────────────────────────
socket.on("qa_update", (data) => {
    if (!data || !data.question || !data.results) return;

    // 1. Update "Last Question"
    lastQuestionEl.innerHTML = `<p class="last-question__text">"${escapeHtml(data.question)}"</p>`;

    // 2. Render note cards
    renderCards(data.results);

    // 3. Push to history
    addToHistory(data.question, data.results);
});

function renderCards(results) {
    qaCardsEl.innerHTML = "";

    if (results.length === 0) {
        qaCardsEl.innerHTML = '<p class="qa-cards__empty">No matching notes found.</p>';
        return;
    }

    results.forEach((r, i) => {
        const card = document.createElement("div");
        card.className = "qa-card";
        card.innerHTML = `
            <div class="qa-card__header">
                <span class="qa-card__badge">📌 Slide ${r.slide_index + 1}: ${escapeHtml(r.heading)}</span>
                <span class="qa-card__score">Score: ${r.score.toFixed(2)}</span>
            </div>
            <p class="qa-card__note">${escapeHtml(r.note_text)}</p>
        `;
        qaCardsEl.appendChild(card);
    });
}

function addToHistory(question, results) {
    // Remove "no history" placeholder
    const placeholder = qaHistoryEl.querySelector(".qa-history__empty");
    if (placeholder) placeholder.remove();

    const entry = document.createElement("div");
    entry.className = "qa-history__entry";

    const notesSummary = results.map(r =>
        `Slide ${r.slide_index + 1} (${r.score.toFixed(2)})`
    ).join(", ");

    entry.innerHTML = `
        <p class="qa-history__question">"${escapeHtml(question)}"</p>
        <p class="qa-history__notes">${notesSummary}</p>
    `;

    // Prepend (newest first)
    qaHistoryEl.prepend(entry);

    // Cap at MAX_HISTORY
    while (qaHistoryEl.children.length > MAX_HISTORY) {
        qaHistoryEl.lastElementChild.remove();
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
```

**How the card-based UI dynamically updates:**

1. `qa_update` event arrives with `{question: str, results: [{slide_index, heading, note_text, score}, ...]}`.
2. `lastQuestionEl.innerHTML` is replaced with the new question text.
3. `renderCards()` clears `#qa-cards` and creates 1–3 `.qa-card` elements with header (slide badge + score) and note text.
4. `addToHistory()` prepends a summary to `#qa-history` (newest first, capped at 10 entries, older entries removed from the DOM).

### 4.5 CSS: Presenter Dashboard Styles (`presenter.css`)

Follows the design system tokens from `style.css` (dark theme, glassmorphism cards, Inter/Outfit fonts).

```css
/* ── Presenter Panel Layout ──────────────────────────────────────────── */

.presenter-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 100px 24px 40px;   /* 100px top to clear topbar */
}

/* ── Status Bar ──────────────────────────────────────────────────────── */

.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    margin-bottom: 28px;
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-danger);       /* Red = disconnected */
    transition: background var(--transition-fast);
}

.status-dot.connected {
    background: var(--accent-success);      /* Green = connected */
}

/* ── Q&A Sections ────────────────────────────────────────────────────── */

.qa-section {
    margin-bottom: 28px;
}

.qa-section__title {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
}

/* ── Last Question ───────────────────────────────────────────────────── */

.last-question {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
}

.last-question__text {
    font-family: var(--font-body);
    font-size: 1.05rem;
    font-style: italic;
    color: var(--accent-primary);
    margin: 0;
}

/* ── Note Cards ──────────────────────────────────────────────────────── */

.qa-cards {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.qa-cards__empty {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-family: var(--font-body);
}

.qa-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    animation: card-fade-in 0.3s ease forwards;
    opacity: 0;
}

.qa-card__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.qa-card__badge {
    font-family: var(--font-heading);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-primary);
}

.qa-card__score {
    font-family: var(--font-body);
    font-size: 0.75rem;
    color: var(--text-muted);
}

.qa-card__note {
    font-family: var(--font-body);
    font-size: 0.95rem;
    color: var(--text-primary);
    line-height: 1.6;
    margin: 0;
}

@keyframes card-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── History ─────────────────────────────────────────────────────────── */

.qa-history {
    max-height: 300px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.qa-history__empty {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-family: var(--font-body);
}

.qa-history__entry {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
}

.qa-history__question {
    font-size: 0.85rem;
    font-style: italic;
    color: var(--text-secondary);
    margin: 0 0 4px;
}

.qa-history__notes {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0;
}
```

---

## 5. Testing Strategy (`tests/test_qa_assistant.py`)

### 5.1 Test Architecture

Tests mock `embeddings.encode` to avoid loading the real sentence-transformers model (~90 MB). Cosine similarity is tested with hand-crafted tensors that simulate real embedding behavior.

```python
# All tests will use this pattern:
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import context_search
from qa_assistant import is_question, search_notes, build_notes_index
```

### 5.2 Sample Slides for Tests

```python
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
```

### 5.3 Question Detection Tests

| # | Test Name | Input | Expected | Rationale |
|---|-----------|-------|----------|-----------|
| 1 | `test_question_what_start` | `"What is the revenue for Q3"` | `True` | WH-word at start |
| 2 | `test_question_how_start` | `"How did we perform this quarter"` | `True` | WH-word at start |
| 3 | `test_question_why_start` | `"Why did expenses increase"` | `True` | WH-word at start |
| 4 | `test_question_where_start` | `"Where are we expanding next year"` | `True` | WH-word at start |
| 5 | `test_question_aux_can` | `"Can you explain the budget allocation"` | `True` | Aux-verb at start |
| 6 | `test_question_aux_is` | `"Is the team growing this year"` | `True` | Aux-verb at start |
| 7 | `test_question_aux_does` | `"Does this include international revenue"` | `True` | Aux-verb at start |
| 8 | `test_question_trailing_mark` | `"the budget is how much?"` | `True` | Trailing `?` |
| 9 | `test_statement_next_slide` | `"Next slide please"` | `False` | Navigation command |
| 10 | `test_statement_declarative` | `"Revenue grew 18 percent year over year"` | `False` | Declarative content |
| 11 | `test_statement_emphasis` | `"This is really important remember this"` | `False` | Emphasis phrase |
| 12 | `test_short_filler_what` | `"What"` | `False` | Too short (< 3 words) |
| 13 | `test_short_filler_how` | `"How so"` | `False` | Too short (< 3 words) |
| 14 | `test_empty_string` | `""` | `False` | Edge case: empty |
| 15 | `test_question_who_start` | `"Who is leading the engineering team"` | `False` → Trick: actually `True` | WH-word at start |

### 5.4 Notes Index & Retrieval Tests

| # | Test Name | Description |
|---|-----------|-------------|
| 16 | `test_build_notes_index_count` | `build_notes_index()` with `SAMPLE_SLIDES` → exactly 4 entries indexed (slide 4 has empty notes, excluded) |
| 17 | `test_build_notes_index_excludes_empty` | Verify slide 4 (`notes: ""`) is NOT in `_note_entries` |
| 18 | `test_build_notes_index_entries_have_correct_fields` | Each entry has `slide_index`, `heading`, `note_text` keys |
| 19 | `test_search_notes_returns_list` | `search_notes("revenue")` returns a `list` |
| 20 | `test_search_notes_max_three` | Even with >3 notes above threshold, result length ≤ 3 |
| 21 | `test_search_notes_results_sorted_by_score` | Results are sorted by descending score |
| 22 | `test_search_notes_result_fields` | Each result dict has `slide_index`, `heading`, `note_text`, `score` |
| 23 | `test_search_notes_empty_index` | With no slides loaded → returns `[]` |
| 24 | `test_search_notes_no_match` | Completely irrelevant query → returns `[]` (all below threshold) |
| 25 | `test_search_notes_relevant_query` | `"What is Q3 revenue"` → top result has `slide_index == 1` (Revenue Overview) |

### 5.5 Integration Tests (Pipeline)

| # | Test Name | Description |
|---|-----------|-------------|
| 26 | `test_question_triggers_qa_update_emit` | Mock transcriber returns a question → `socketio.emit("qa_update", ...)` is called |
| 27 | `test_statement_does_not_trigger_qa_update` | Mock transcriber returns a statement → `qa_update` is NOT emitted |
| 28 | `test_question_does_not_block_intent_classification` | Question detected → `classify_intent()` is still called (non-blocking) |
| 29 | `test_qa_update_payload_structure` | Verify emitted payload has `question` (str) and `results` (list of dicts) |
| 30 | `test_presenter_route_returns_200` | `GET /presenter` → status 200 |

### 5.6 Test Count

| Category | Count |
|----------|-------|
| Existing Phase 1–6 tests | 99 |
| New: Question detection (tests 1–15) | 15 |
| New: Notes index & retrieval (tests 16–25) | 10 |
| New: Pipeline integration (tests 26–30) | 5 |
| **Total** | **129** |

---

## 6. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/qa_assistant.py` | **CREATE** | Question detection (regex), notes indexing (embeddings), search_notes (cosine similarity) |
| `backend/app.py` | **EDIT** | Import `qa_assistant`, add `/presenter` route, add `build_notes_index()` calls at 3 index-rebuild sites, add question interception in `_process_speech_buffer()` |
| `frontend/presenter.html` | **CREATE** | Presenter Panel page — status bar, last question, note cards, history |
| `frontend/js/presenter.js` | **CREATE** | WebSocket listener for `qa_update`, dynamic card rendering, history management |
| `frontend/css/presenter.css` | **CREATE** | Dashboard layout, card styles, status indicator, history scroll |
| `tests/test_qa_assistant.py` | **CREATE** | 30 tests: question detection (15), notes retrieval (10), pipeline integration (5) |

**Files unchanged:** All existing backend modules (`transcriber.py`, `intent_classifier.py`, `embeddings.py`, `context_search.py`, `keyword_highlighter.py`, `vad_engine.py`, `slide_loader.py`, `config.py`), all existing frontend files (`index.html`, `app.js`, `presentation.css`), and all existing test files.

---

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| **False positive questions** — "What a great quarter" starts with "what" | The 3-word minimum filter helps. "What a" is only 2 meaningful words in "what a great quarter" but still 4 words total, so it passes the length check. However, WH-exclamations like "What a great quarter" will trigger a false positive. **Accepted trade-off:** A false positive merely shows notes on the presenter panel — no user-facing disruption. The presenter can simply ignore irrelevant suggestions. |
| **Empty notes** — Most slides have `notes: ""` | `_extract_notes()` skips empty notes. If zero notes exist, `search_notes()` returns `[]` and no `qa_update` is emitted. The presenter panel shows "No matching notes found." No crash. |
| **Embedding model not yet loaded** when first question arrives | `embeddings.encode()` calls `get_embedding_model()` which lazy-loads the model on first use. If `build_notes_index()` already ran at startup, the model is already loaded. If not (e.g., no slides.json), the first `encode()` call will load it. ~2s delay on first call, then instant. |
| **Presenter panel open in multiple tabs** | Socket.IO broadcasts `qa_update` to all connected clients. Multiple `/presenter` tabs all receive the same event and render the same cards. No conflict. The main `index.html` page also receives `qa_update` but has no handler for it — safely ignored. |
| **Existing 99 tests break** | `qa_assistant` is only imported in `app.py` at module level. Existing tests that `@patch("app.context_search")` will mock context_search but `qa_assistant` calls `context_search._slides` directly. However, the question interception in `_process_speech_buffer()` can be handled by also mocking `app.qa_assistant` — same pattern as the existing `@patch("app.fuzzy_match_current_slide")`. The existing tests do NOT call `qa_assistant` functions, so they remain unaffected. |
| **`qa_update` fires too frequently** | Questions tend to be full utterances (3+ words, interrogative form). VAD already segments speech into complete utterances. Rapid-fire questions are unlikely in practice. No cooldown needed. |

---

*Awaiting strict approval before generating Python/HTML code.*
