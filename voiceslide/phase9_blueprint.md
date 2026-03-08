# Phase 9 — Speech Analytics Dashboard

## Problem Statement

After a presentation, the speaker has no objective data about their speaking performance. Specifically:

1. **No filler word tracking** — the speaker has no awareness of how often they say "um", "uh", "like", "you know", "so", "basically", "actually", "right", "I mean", etc.
2. **No pace measurement** — there is no words-per-minute (WPM) metric to tell the speaker if they're rushing or dragging.
3. **No sentiment analysis** — there is no way to gauge whether the speaker's language was confident, positive, neutral, or negative across the presentation timeline.

**Solution:** Build a backend `analytics.py` module that silently tracks all three metrics during live speech, and a post-presentation `/analytics` dashboard that visualizes the data using Chart.js.

---

## 1. Dependencies & Models

### 1.1 Python Packages

| Package | Status | Purpose |
|---------|--------|---------|
| `vaderSentiment>=3.3.2` | ❌ **Needs install** | Lexicon-based sentiment scoring (no model download, no GPU, ~1 MB) |

Add to `requirements.txt`:
```
# Phase 9 Dependencies — Speech Analytics (Sentiment Analysis)
vaderSentiment>=3.3.2
```

**Why VADER?**
- **Zero model loading** — VADER is a rule-based lexicon. It imports instantly and scores text in microseconds. No transformer model, no GPU, no RAM impact.
- **Designed for short texts** — VADER was built for social media and short utterances, which perfectly matches our STT segments (1–3 sentences each).
- **Compound score** — returns a single float in [-1.0, +1.0] that maps directly to negative → neutral → positive. Ideal for a time-series chart.

### 1.2 Frontend Library

| Library | Delivery | Purpose |
|---------|----------|---------|
| `Chart.js v4` | CDN (`https://cdn.jsdelivr.net/npm/chart.js@4`) | Line chart (sentiment over time), bar chart (filler word breakdown), stat blocks (WPM) |

No npm install. Loaded via `<script>` tag in `analytics.html`, matching the existing CDN pattern for reveal.js and Socket.IO.

### 1.3 RAM Budget Impact

| State | Models in RAM | Estimated RAM |
|-------|--------------|---------------|
| Phase 8 (current) | faster-whisper base + all-MiniLM-L6-v2 + Silero VAD | ~1.2 GB |
| Phase 9 (analytics active) | Same — VADER is a dict lookup, not a model | ~1.2 GB + ~0 |

**Impact: Zero.** VADER's lexicon is a Python dict loaded once on import (~300 KB). The analytics tracker stores plain Python lists/dicts of metrics — negligible memory even for a 60-minute presentation.

---

## 2. Backend Tracking Logic (`analytics.py`)

### 2.1 Module Responsibilities

```
analytics.py
├── AnalyticsTracker (class)
│   ├── __init__()              — Initialize empty session state
│   ├── reset()                 — Clear all metrics for a new session
│   ├── record_segment(text)    — Process one transcript segment: count fillers, update WPM, score sentiment
│   ├── get_summary()           — Return the full analytics payload as a dict
│   └── Internal state:
│       ├── _segments: list[dict]           — [{text, timestamp, word_count, filler_count, fillers_found, sentiment}, ...]
│       ├── _session_start: float|None      — time.time() when first segment arrives
│       ├── _total_words: int               — Running word count
│       └── _total_fillers: int             — Running filler count
├── _FILLER_RE (compiled regex)             — Pattern matching filler words/phrases
├── FILLER_LIST (list)                      — Canonical list of target fillers
└── _analyzer (SentimentIntensityAnalyzer)  — VADER singleton, created at module import
```

### 2.2 Module-Level Singleton

```python
"""
VoiceSlide — Speech Analytics Tracker (Phase 9)
Silently records filler words, speaking pace, and sentiment for each
transcript segment during a live presentation session.
"""

import logging
import re
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("voiceslide.analytics")

# VADER is a lexicon lookup — instantiate once at import time (~1ms, ~300 KB)
_analyzer = SentimentIntensityAnalyzer()
```

The module exposes a single `tracker` instance (module-level singleton). This matches the project pattern: `context_search` uses module-level `_slides` and `_embeddings`, `qa_assistant` uses module-level `_note_entries` and `_note_embeddings`. Similarly, `analytics.py` uses a module-level `tracker` object.

### 2.3 Filler Word Detection

#### Target List

```python
FILLER_LIST = [
    "um", "uh", "uhm", "umm",          # Hesitation sounds
    "like",                              # Discourse filler
    "you know",                          # Phrase filler (2 words)
    "i mean",                            # Phrase filler (2 words)
    "sort of", "kind of",               # Hedging phrases (2 words)
    "basically", "actually", "literally",# Overused adverbs
    "right", "okay", "so",              # Discourse markers (start-of-sentence)
]
```

#### Regex Approach

Multi-word fillers like "you know" and "sort of" must be matched as phrases, not individual words. Single-word fillers must be matched as whole words to avoid false positives (e.g., "like" inside "likelihood").

```python
_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in sorted(FILLER_LIST, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
```

**Key design decisions:**

1. **Sort by length descending** — ensures `"you know"` is matched before `"you"` would be (though `"you"` is not in the list, this prevents substring issues if the list grows).
2. **`\b` word boundaries** — per the copilot instructions, `\b` fires at `\w↔\W` transitions. All our fillers consist of `\w` characters (letters), so `\b` correctly prevents matching "like" inside "likelihood" because there's no `\w↔\W` transition between "like" and "lihood" — both sides are `\w`. The boundary fires before the "l" of "like" (space→letter = `\W↔\w`) and after the "e" of "like" only if the next character is `\W` (space, punctuation, or end-of-string).
3. **Case-insensitive** — STT output casing is unpredictable.

#### Counting Function

```python
def _count_fillers(text: str) -> dict:
    """Count filler word occurrences in a text segment.

    Returns:
        {"total": int, "breakdown": {"um": 2, "like": 1, ...}}
    """
    matches = _FILLER_RE.findall(text.lower())
    breakdown = {}
    for m in matches:
        key = m.lower()
        breakdown[key] = breakdown.get(key, 0) + 1
    return {"total": len(matches), "breakdown": breakdown}
```

### 2.4 WPM Calculation

```python
def _count_words(text: str) -> int:
    """Count words in a text segment (simple whitespace split)."""
    return len(text.split())
```

WPM is computed in `get_summary()` from the running totals:

```python
elapsed_minutes = (time.time() - self._session_start) / 60.0
wpm = self._total_words / elapsed_minutes if elapsed_minutes > 0 else 0.0
```

**Why not per-segment WPM?** Individual segments vary wildly in length (2–20 words) and duration (VAD timing is not exact). The **session-average WPM** is a much more meaningful metric for the speaker. However, we also store per-segment word counts to enable a "rolling WPM" chart (computed on the frontend using a sliding window).

### 2.5 Sentiment Scoring with VADER

```python
def _score_sentiment(text: str) -> dict:
    """Score sentiment of a text segment using VADER.

    Returns the full VADER dict:
        {"neg": float, "neu": float, "pos": float, "compound": float}

    The 'compound' score is the primary metric:
        -1.0 (most negative) → 0.0 (neutral) → +1.0 (most positive)
    """
    return _analyzer.polarity_scores(text)
```

The `compound` score is used for the time-series chart. The `pos`/`neg`/`neu` breakdown can be shown in tooltips.

### 2.6 AnalyticsTracker Class

```python
class AnalyticsTracker:
    """Tracks speech analytics for a single presentation session."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all metrics for a new session."""
        self._segments = []
        self._session_start = None
        self._total_words = 0
        self._total_fillers = 0

    def record_segment(self, text: str) -> None:
        """Record metrics for one transcript segment.

        Called once per utterance from _process_speech_buffer().
        """
        if not text or not text.strip():
            return

        now = time.time()
        if self._session_start is None:
            self._session_start = now

        word_count = _count_words(text)
        filler_result = _count_fillers(text)
        sentiment = _score_sentiment(text)

        self._total_words += word_count
        self._total_fillers += filler_result["total"]

        self._segments.append({
            "timestamp": round(now - self._session_start, 2),
            "text": text,
            "word_count": word_count,
            "filler_count": filler_result["total"],
            "fillers_found": filler_result["breakdown"],
            "sentiment": sentiment["compound"],
        })

    def get_summary(self) -> dict:
        """Return the full analytics payload for the dashboard.

        Returns:
            {
                "session_duration": float (seconds),
                "total_words": int,
                "total_fillers": int,
                "filler_ratio": float (fillers per 100 words),
                "avg_wpm": float,
                "filler_breakdown": {"um": 5, "like": 3, ...},
                "segments": [
                    {"timestamp": float, "text": str, "word_count": int,
                     "filler_count": int, "sentiment": float},
                    ...
                ],
            }
        """
        if not self._segments:
            return {
                "session_duration": 0.0,
                "total_words": 0,
                "total_fillers": 0,
                "filler_ratio": 0.0,
                "avg_wpm": 0.0,
                "filler_breakdown": {},
                "segments": [],
            }

        elapsed = time.time() - self._session_start
        elapsed_min = elapsed / 60.0

        # Aggregate filler breakdown across all segments
        agg_breakdown = {}
        for seg in self._segments:
            for filler, count in seg["fillers_found"].items():
                agg_breakdown[filler] = agg_breakdown.get(filler, 0) + count

        return {
            "session_duration": round(elapsed, 2),
            "total_words": self._total_words,
            "total_fillers": self._total_fillers,
            "filler_ratio": round(
                (self._total_fillers / self._total_words * 100) if self._total_words > 0 else 0.0, 2
            ),
            "avg_wpm": round(self._total_words / elapsed_min if elapsed_min > 0 else 0.0, 1),
            "filler_breakdown": agg_breakdown,
            "segments": [
                {
                    "timestamp": s["timestamp"],
                    "text": s["text"],
                    "word_count": s["word_count"],
                    "filler_count": s["filler_count"],
                    "sentiment": s["sentiment"],
                }
                for s in self._segments
            ],
        }


# Module-level singleton
tracker = AnalyticsTracker()
```

### 2.7 Why a Class, Not Module-Level Variables?

`context_search` and `qa_assistant` use module-level globals because their state is simple (a list + a tensor). The analytics tracker has **correlated** state: `_segments`, `_session_start`, `_total_words`, and `_total_fillers` must be reset atomically. A class with a `reset()` method is cleaner and less error-prone than four separate `global` declarations. The module still exposes a single `tracker` instance, consistent with the project's singleton-per-module pattern.

---

## 3. Backend Integration (`app.py`)

### 3.1 New Import

```python
import analytics
```

Added alongside the existing Phase 5–8 imports (after line 50, alongside `import qa_assistant`).

### 3.2 Pipeline Integration Point

The analytics call is inserted in `_process_speech_buffer()` **after** the transcript emit (line 271) and **before** the Phase 8 Q&A check (line 273). This position ensures:
- The transcript is already validated (non-empty `transcribed_text`).
- The analytics recording is fire-and-forget — no return value, no branching.
- It does not interfere with Q&A detection or intent classification.

```python
    socketio.emit("transcript", result)                    # line 271 (existing)

    # ── Phase 9: Record speech analytics ─────────────────────────────
    analytics.tracker.record_segment(transcribed_text)

    # ── Phase 8: Q&A — detect questions, search notes ────────────── # line 273 (existing)
```

**Why this position?**
1. **After transcript emit** — the text is confirmed non-empty and displayable.
2. **Before Q&A and intent** — analytics is purely observational. It must not delay or influence any downstream logic.
3. **Single line, no branching** — `record_segment()` always succeeds (even on empty text it returns immediately). Zero risk of breaking the pipeline.

### 3.3 New Flask Route: `/analytics`

```python
@app.route("/analytics")
def analytics_page():
    """Serve the Speech Analytics Dashboard page."""
    return send_from_directory(config.FRONTEND_DIR, "analytics.html")
```

Placed alongside the existing page routes (`/`, `/upload`, `/presenter`) around line 82.

### 3.4 New API Route: `/api/analytics`

```python
@app.route("/api/analytics", methods=["GET"])
def api_get_analytics():
    """Return the current session's speech analytics as JSON."""
    return jsonify(analytics.tracker.get_summary()), 200
```

Placed alongside the existing API routes (`/api/slides`, `/api/upload-pptx`, `/api/save-slides`).

### 3.5 Session Reset via WebSocket Event

```python
@socketio.on("reset_analytics")
def handle_reset_analytics():
    """Reset analytics tracker for a new presentation session."""
    analytics.tracker.reset()
    logger.info("Analytics session reset.")
    socketio.emit("analytics_reset")
```

This event is triggered by a "Reset Session" button on the analytics dashboard. The `analytics_reset` confirmation event lets the frontend clear its charts.

### 3.6 Pipeline Flow with Phase 9

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
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
    socketio.emit           analytics.tracker      is_question()
    ("transcript")          .record_segment()           │
                           (silent, non-blocking)  ┌────┴────┐
                                   │               ▼         ▼
                                   ▼          search_notes  (skip)
                            Stores in              │
                            _segments         socketio.emit
                            (filler count,    ("qa_update")
                             word count,           │
                             sentiment)            ▼
                                   │         /presenter
                                   ▼
                      classify_intent()
                             │
                       ┌─────┴─────┐
                       ▼           ▼
                  nav_command  fuzzy_match /
                               context_search

                 ═══════════════════════════════════
                 GET /api/analytics  (post-session)
                           │
                           ▼
                 ┌─────────────────┐
                 │   /analytics    │
                 │   dashboard     │
                 └─────────────────┘
```

---

## 4. Session Lifecycle

### 4.1 When Does a Session Start?

The session starts **lazily** — on the first call to `record_segment()` after a reset. The `_session_start` timestamp is set to `time.time()` on that first call. This means:
- No explicit "start" event is needed from the frontend.
- The timer begins when the speaker first speaks, not when they press the mic button. This gives a more accurate WPM since it excludes silence-before-first-utterance.

### 4.2 When Does a Session End?

The session has **no explicit end**. The analytics data accumulates as long as segments arrive. The speaker views the dashboard **after** the presentation by navigating to `/analytics`, which calls `GET /api/analytics` to fetch the accumulated data.

### 4.3 How to Reset for a New Session?

Two mechanisms:
1. **Manual reset** — the analytics dashboard has a "Reset Session" button that emits `reset_analytics` via WebSocket. This calls `tracker.reset()`, clearing all data.
2. **Automatic reset** — NOT implemented in Phase 9. In a future phase, this could be tied to a "new presentation" event. For now, manual reset is sufficient and avoids accidental data loss.

### 4.4 What About WebSocket Disconnect?

`handle_disconnect()` already clears the speech buffer and resets VAD state. We do **NOT** reset analytics on disconnect — the speaker may accidentally close the tab and reopen it. Analytics data persists in memory until explicitly reset.

---

## 5. Frontend UI

### 5.1 New Files

| File | Purpose |
|------|---------|
| `frontend/analytics.html` | Speech Analytics Dashboard page structure |
| `frontend/js/analytics.js` | Fetch `/api/analytics`, render Chart.js charts, handle reset |
| `frontend/css/analytics.css` | Dashboard layout, stat blocks, chart containers |

### 5.2 Dashboard Layout (`analytics.html`)

```
┌─────────────────────────────────────────────────────────────────────┐
│  VoiceSlide                           [Back to Presentation]        │
│  ─ Speech Analytics ─                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ Stat Blocks ─────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ ⏱️  WPM  │  │ 🗣️ Words │  │ 🔸 Fillers│  │ 📊 Filler % │  │  │
│  │  │   142    │  │   1,284  │  │    47     │  │    3.66%     │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Sentiment Over Time ─────────────────────────────────────────┐  │
│  │                                                               │  │
│  │    ~~~~ Line Chart (compound score -1..+1 vs timestamp) ~~~~  │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Filler Word Breakdown ───────────────────────────────────────┐  │
│  │                                                               │  │
│  │    ████ Bar Chart (filler word → count, sorted desc) ████     │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ Actions ─────────────────────────────────────────────────────┐  │
│  │  [🔄 Refresh Data]     [🗑️ Reset Session]                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key UI elements:**

| Element | ID | Purpose |
|---------|----|---------|
| WPM stat block | `#stat-wpm` | Average words per minute |
| Words stat block | `#stat-words` | Total word count |
| Fillers stat block | `#stat-fillers` | Total filler count |
| Filler ratio stat block | `#stat-ratio` | Fillers per 100 words (percentage) |
| Sentiment chart | `#sentiment-chart` | Line chart: compound score over time |
| Filler chart | `#filler-chart` | Horizontal bar chart: breakdown by filler word |
| Refresh button | `#btn-refresh` | Re-fetch `/api/analytics` and redraw charts |
| Reset button | `#btn-reset` | Emit `reset_analytics` and clear dashboard |

### 5.3 HTML Structure (`analytics.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VoiceSlide — Speech Analytics</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/css/style.css" />
  <link rel="stylesheet" href="/css/analytics.css" />
</head>
<body>
  <header class="topbar">
    <div class="topbar__logo">VoiceSlide</div>
    <span class="topbar__subtitle">Speech Analytics</span>
    <nav class="topbar__nav">
      <a href="/" class="btn-text">Back to Presentation</a>
    </nav>
  </header>

  <main class="analytics-container">
    <!-- Stat Blocks -->
    <section class="stats-row">
      <div class="stat-block" id="stat-wpm">
        <span class="stat-block__icon">⏱️</span>
        <span class="stat-block__value">—</span>
        <span class="stat-block__label">Avg WPM</span>
      </div>
      <div class="stat-block" id="stat-words">
        <span class="stat-block__icon">🗣️</span>
        <span class="stat-block__value">—</span>
        <span class="stat-block__label">Total Words</span>
      </div>
      <div class="stat-block" id="stat-fillers">
        <span class="stat-block__icon">🔸</span>
        <span class="stat-block__value">—</span>
        <span class="stat-block__label">Filler Words</span>
      </div>
      <div class="stat-block" id="stat-ratio">
        <span class="stat-block__icon">📊</span>
        <span class="stat-block__value">—</span>
        <span class="stat-block__label">Filler Ratio</span>
      </div>
    </section>

    <!-- Sentiment Over Time -->
    <section class="chart-section">
      <h2 class="chart-section__title">Sentiment Over Time</h2>
      <div class="chart-wrapper">
        <canvas id="sentiment-chart"></canvas>
      </div>
    </section>

    <!-- Filler Word Breakdown -->
    <section class="chart-section">
      <h2 class="chart-section__title">Filler Word Breakdown</h2>
      <div class="chart-wrapper chart-wrapper--short">
        <canvas id="filler-chart"></canvas>
      </div>
    </section>

    <!-- Actions -->
    <section class="actions-row">
      <button id="btn-refresh" class="btn-action">🔄 Refresh Data</button>
      <button id="btn-reset" class="btn-action btn-action--danger">🗑️ Reset Session</button>
    </section>

    <!-- Empty State -->
    <div id="empty-state" class="empty-state" style="display:none;">
      <p>No speech data yet. Start a presentation and speak to generate analytics.</p>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <script src="/js/analytics.js"></script>
</body>
</html>
```

### 5.4 JavaScript: Data Fetch & Chart Rendering (`analytics.js`)

```javascript
"use strict";

// ── DOM References ──────────────────────────────────────────────────────
const statWpm     = document.querySelector("#stat-wpm .stat-block__value");
const statWords   = document.querySelector("#stat-words .stat-block__value");
const statFillers = document.querySelector("#stat-fillers .stat-block__value");
const statRatio   = document.querySelector("#stat-ratio .stat-block__value");
const btnRefresh  = document.getElementById("btn-refresh");
const btnReset    = document.getElementById("btn-reset");
const emptyState  = document.getElementById("empty-state");

let sentimentChart = null;
let fillerChart = null;
const socket = io();

// ── Chart.js Global Config (dark theme) ────────────────────────────────
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "rgba(255,255,255,0.08)";
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

// ── Fetch & Render ─────────────────────────────────────────────────────
async function loadAnalytics() {
    try {
        const res = await fetch("/api/analytics");
        const data = await res.json();
        renderDashboard(data);
    } catch (err) {
        console.error("[Analytics] Failed to fetch data:", err);
    }
}

function renderDashboard(data) {
    const hasData = data.segments && data.segments.length > 0;
    emptyState.style.display = hasData ? "none" : "block";

    // Stat blocks
    statWpm.textContent     = hasData ? data.avg_wpm.toFixed(0) : "—";
    statWords.textContent   = hasData ? data.total_words.toLocaleString() : "—";
    statFillers.textContent = hasData ? data.total_fillers.toLocaleString() : "—";
    statRatio.textContent   = hasData ? data.filler_ratio.toFixed(2) + "%" : "—";

    // Sentiment chart
    renderSentimentChart(data.segments || []);

    // Filler chart
    renderFillerChart(data.filler_breakdown || {});
}

function renderSentimentChart(segments) {
    const ctx = document.getElementById("sentiment-chart");

    const labels = segments.map(s => {
        const mins = Math.floor(s.timestamp / 60);
        const secs = Math.floor(s.timestamp % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    });
    const values = segments.map(s => s.sentiment);

    if (sentimentChart) sentimentChart.destroy();

    sentimentChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Sentiment (compound)",
                data: values,
                borderColor: "#00c8ff",
                backgroundColor: "rgba(0, 200, 255, 0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    title: { display: true, text: "Sentiment" },
                },
                x: {
                    title: { display: true, text: "Time" },
                },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        afterBody: (items) => {
                            const idx = items[0]?.dataIndex;
                            if (idx !== undefined && segments[idx]) {
                                return `"${segments[idx].text}"`;
                            }
                        },
                    },
                },
            },
        },
    });
}

function renderFillerChart(breakdown) {
    const ctx = document.getElementById("filler-chart");

    // Sort by count descending
    const sorted = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([word]) => word);
    const values = sorted.map(([, count]) => count);

    if (fillerChart) fillerChart.destroy();

    fillerChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Count",
                data: values,
                backgroundColor: "rgba(124, 58, 237, 0.6)",
                borderColor: "#7c3aed",
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { stepSize: 1 },
                    title: { display: true, text: "Occurrences" },
                },
            },
        },
    });
}

// ── Actions ────────────────────────────────────────────────────────────
btnRefresh.addEventListener("click", loadAnalytics);

btnReset.addEventListener("click", () => {
    if (confirm("Reset all analytics data for this session?")) {
        socket.emit("reset_analytics");
    }
});

socket.on("analytics_reset", () => {
    loadAnalytics();   // Re-render with empty data
});

// ── Initial Load ───────────────────────────────────────────────────────
loadAnalytics();
```

### 5.5 CSS: Analytics Dashboard Styles (`analytics.css`)

```css
/* ── Analytics Dashboard Layout ──────────────────────────────────────── */

.analytics-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 100px 24px 40px;
}

/* ── Stat Blocks Row ─────────────────────────────────────────────────── */

.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

.stat-block {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    text-align: center;
}

.stat-block__icon {
    font-size: 1.4rem;
}

.stat-block__value {
    font-family: var(--font-heading);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-primary);
}

.stat-block__label {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* ── Chart Sections ──────────────────────────────────────────────────── */

.chart-section {
    margin-bottom: 32px;
}

.chart-section__title {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
}

.chart-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 20px;
    height: 300px;
    position: relative;
}

.chart-wrapper--short {
    height: 220px;
}

/* ── Action Buttons ──────────────────────────────────────────────────── */

.actions-row {
    display: flex;
    gap: 12px;
    margin-bottom: 32px;
}

.btn-action {
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 500;
    padding: 10px 20px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--glass-border);
    background: var(--bg-card);
    color: var(--text-primary);
    cursor: pointer;
    transition: background var(--transition-fast), border-color var(--transition-fast);
}

.btn-action:hover {
    background: var(--bg-surface);
    border-color: var(--accent-primary);
}

.btn-action--danger:hover {
    border-color: var(--accent-danger);
    color: var(--accent-danger);
}

/* ── Empty State ─────────────────────────────────────────────────────── */

.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-muted);
    font-family: var(--font-body);
    font-size: 0.95rem;
}

/* ── Responsive ──────────────────────────────────────────────────────── */

@media (max-width: 640px) {
    .stats-row {
        grid-template-columns: repeat(2, 1fr);
    }
}
```

---

## 6. Testing Strategy (`tests/test_analytics.py`)

### 6.1 Test Architecture

Tests import `analytics` directly and call functions/methods on the `AnalyticsTracker` class. No model mocking needed — VADER is a lightweight lexicon lookup that runs instantly. No external dependencies to mock.

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from analytics import (
    AnalyticsTracker, _count_fillers, _count_words, _score_sentiment,
    FILLER_LIST, _FILLER_RE,
)
import app as voice_app
```

### 6.2 Filler Word Detection Tests

| # | Test Name | Input | Expected | Rationale |
|---|-----------|-------|----------|-----------|
| 1 | `test_filler_um_detected` | `"I think um we should proceed"` | `total=1, breakdown={"um": 1}` | Basic hesitation filler |
| 2 | `test_filler_uh_detected` | `"Revenue uh grew this quarter"` | `total=1, breakdown={"uh": 1}` | Basic hesitation filler |
| 3 | `test_filler_like_detected` | `"It was like really impressive"` | `total=1, breakdown={"like": 1}` | Discourse filler |
| 4 | `test_filler_like_not_in_likelihood` | `"The likelihood of success is high"` | `total=0` | `\b` prevents false match inside "likelihood" |
| 5 | `test_filler_you_know_phrase` | `"We should you know focus more"` | `total=1, breakdown={"you know": 1}` | Multi-word phrase filler |
| 6 | `test_filler_multiple_types` | `"Um I think like you know it was basically fine"` | `total=4` (um, like, you know, basically) | Multiple different fillers |
| 7 | `test_filler_repeated_same` | `"Um um um let me think"` | `total=3, breakdown={"um": 3}` | Same filler repeated |
| 8 | `test_filler_case_insensitive` | `"UM Like BASICALLY"` | `total=3` | Case shouldn't matter |
| 9 | `test_filler_none_in_clean_speech` | `"Revenue grew eighteen percent year over year"` | `total=0` | No fillers in clean text |
| 10 | `test_filler_empty_string` | `""` | `total=0` | Edge case |
| 11 | `test_filler_sort_of_phrase` | `"It was sort of expected"` | `total=1, breakdown={"sort of": 1}` | Multi-word hedge filler |
| 12 | `test_filler_so_at_start` | `"So we decided to expand"` | `total=1, breakdown={"so": 1}` | Discourse marker |

### 6.3 Word Count Tests

| # | Test Name | Input | Expected | Rationale |
|---|-----------|-------|----------|-----------|
| 13 | `test_word_count_normal` | `"Revenue grew eighteen percent"` | `4` | Basic word count |
| 14 | `test_word_count_empty` | `""` | `0` | Edge case |
| 15 | `test_word_count_single_word` | `"Hello"` | `1` | Single word |

### 6.4 Sentiment Scoring Tests

| # | Test Name | Input | Expected | Rationale |
|---|-----------|-------|----------|-----------|
| 16 | `test_sentiment_positive` | `"This is a great achievement and we are very proud"` | `compound > 0.3` | Clearly positive text |
| 17 | `test_sentiment_negative` | `"This is terrible and we failed badly"` | `compound < -0.3` | Clearly negative text |
| 18 | `test_sentiment_neutral` | `"The meeting is at three o'clock"` | `-0.3 ≤ compound ≤ 0.3` | Factual, neutral text |
| 19 | `test_sentiment_returns_compound` | `"Hello world"` | `"compound" in result` | Verify dict structure |
| 20 | `test_sentiment_empty_string` | `""` | `compound == 0.0` | VADER returns 0 for empty |

### 6.5 AnalyticsTracker Tests

| # | Test Name | Description |
|---|-----------|-------------|
| 21 | `test_tracker_initial_state` | Fresh tracker → `get_summary()` returns all zeros, empty segments |
| 22 | `test_tracker_record_one_segment` | Record `"Hello everyone"` → `total_words == 2`, `segments` length 1 |
| 23 | `test_tracker_record_multiple_segments` | Record 3 segments → `segments` length 3, word count summed correctly |
| 24 | `test_tracker_filler_accumulation` | Record `"Um like we should"` then `"Uh basically yes"` → `total_fillers == 4` |
| 25 | `test_tracker_wpm_calculation` | Record segments over known elapsed time → `avg_wpm` within expected range |
| 26 | `test_tracker_filler_ratio` | Record 10 words with 2 fillers → `filler_ratio == 20.0` |
| 27 | `test_tracker_reset_clears_all` | Record segments → `reset()` → `get_summary()` returns all zeros |
| 28 | `test_tracker_segment_has_timestamp` | First segment → `timestamp == 0.0` (or very close to 0) |
| 29 | `test_tracker_segment_has_sentiment` | Record `"Great work everyone"` → segment `sentiment > 0` |
| 30 | `test_tracker_filler_breakdown_aggregated` | Record multiple segments → `filler_breakdown` aggregates across all |
| 31 | `test_tracker_empty_text_ignored` | `record_segment("")` → segments stays empty |
| 32 | `test_tracker_summary_structure` | `get_summary()` has keys: `session_duration`, `total_words`, `total_fillers`, `filler_ratio`, `avg_wpm`, `filler_breakdown`, `segments` |

### 6.6 Integration Tests (Pipeline)

| # | Test Name | Description |
|---|-----------|-------------|
| 33 | `test_analytics_route_returns_200` | `GET /analytics` → status 200 |
| 34 | `test_api_analytics_returns_json` | `GET /api/analytics` → status 200, response is valid JSON with expected keys |
| 35 | `test_pipeline_calls_record_segment` | Mock analytics.tracker → verify `record_segment()` called with transcribed text during `_process_speech_buffer()` |
| 36 | `test_pipeline_analytics_does_not_block_qa` | Verify Q&A interception still runs after analytics recording |
| 37 | `test_pipeline_analytics_does_not_block_intent` | Verify `classify_intent()` still runs after analytics recording |

### 6.7 Test Count

| Category | Count |
|----------|-------|
| Existing Phase 1–8 tests | 135 |
| New: Filler detection (tests 1–12) | 12 |
| New: Word count (tests 13–15) | 3 |
| New: Sentiment scoring (tests 16–20) | 5 |
| New: AnalyticsTracker (tests 21–32) | 12 |
| New: Pipeline integration (tests 33–37) | 5 |
| **Total** | **172** |

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/analytics.py` | **CREATE** | `AnalyticsTracker` class, filler regex, VADER sentiment, WPM math |
| `backend/app.py` | **EDIT** | Import `analytics`, add `tracker.record_segment()` in pipeline, add `/analytics` page route, add `/api/analytics` GET route, add `reset_analytics` WebSocket handler |
| `backend/requirements.txt` | **EDIT** | Add `vaderSentiment>=3.3.2` |
| `frontend/analytics.html` | **CREATE** | Dashboard page — stat blocks, chart canvases, action buttons |
| `frontend/js/analytics.js` | **CREATE** | Fetch analytics JSON, render Chart.js charts, reset handler |
| `frontend/css/analytics.css` | **CREATE** | Dashboard layout, stat blocks, chart containers, responsive grid |
| `tests/test_analytics.py` | **CREATE** | 37 tests: filler detection (12), word count (3), sentiment (5), tracker class (12), pipeline integration (5) |

**Files unchanged:** All existing backend modules, all existing frontend files, all existing test files.

---

## 8. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| **False positive fillers** — "like" in "I would like to present" | `\b` word boundaries ensure whole-word matching. "like" as a verb is indistinguishable from "like" as a filler without syntactic parsing. **Accepted trade-off:** Over-counting "like" is preferable to missing it. The dashboard shows a breakdown so the speaker can judge. |
| **VADER inaccurate on domain-specific language** | VADER is tuned for general English. Technical jargon ("API", "latency", "throughput") is neutral in VADER, which is correct — these words carry no sentiment. The compound score reflects the speaker's tone words, not their topic. |
| **WPM skewed by long pauses** | `_session_start` is set on the first `record_segment()` call, not on mic activation. Long silence between utterances still inflates elapsed time, deflating WPM. This is intentional — a speaker who pauses often genuinely has a lower effective WPM. |
| **Memory growth on very long sessions** | Each segment stores ~200 bytes of metadata. A 60-minute session at 4 utterances/minute = 240 segments × 200 bytes = ~48 KB. Negligible. |
| **Chart.js bundle size** | Chart.js v4 is ~200 KB gzipped via CDN. Loaded only on the `/analytics` page, not the main presentation. No impact on presentation performance. |
| **Existing 135 tests break** | `analytics.record_segment()` is called in `_process_speech_buffer()` after the transcript emit. Existing tests that mock `app.socketio` and `app.transcribe_chunk` do NOT mock `app.analytics` — but `record_segment()` has no return value and no side effects visible to the pipeline. Existing tests may trigger it but it simply appends to an internal list. Tests that use `@patch("app.context_search")` don't interfere because `analytics.py` does not import `context_search`. However, to be safe, integration tests that check specific `socketio.emit` calls should also patch `app.analytics` to isolate behavior. We will verify all 135 existing tests still pass after the edit. |

---

*Awaiting strict approval before generating Python/HTML code.*
