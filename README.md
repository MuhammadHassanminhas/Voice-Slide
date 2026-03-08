

<h1 align="center">VoiceSlide</h1>

<p align="center">
  <strong>🎙️ NLP-powered presentations with voice navigation, live highlighting, and speech analytics.</strong>
</p>

<p align="center">
  <a href="[INSERT_REPO_URL]/actions"><img src="https://img.shields.io/github/actions/workflow/status/[YOUR_GITHUB_USERNAME]/voiceslide/ci.yml?branch=main&style=flat-square&logo=github&label=build" alt="Build Status" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/flask-3.x-000000?style=flat-square&logo=flask" alt="Flask 3.x" />
  <img src="https://img.shields.io/badge/license-[INSERT_LICENSE]-blue?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/whisper-faster--whisper-ff6f00?style=flat-square" alt="faster-whisper" />
  <img src="https://img.shields.io/badge/realtime-Socket.IO-010101?style=flat-square&logo=socket.io" alt="Socket.IO" />
  <img src="https://img.shields.io/badge/slides-reveal.js%205-f7df1e?style=flat-square" alt="reveal.js" />
</p>
<p align="center">
 <img width="1919" height="920" alt="image" src="https://github.com/user-attachments/assets/aea8c765-4ab5-47c1-acd6-3e9962598766" />

</p>
---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact & Acknowledgements](#-contact--acknowledgements)

---

## 🔭 Overview

**VoiceSlide** is a full-stack presentation platform that lets you **control slides with your voice**. It uses on-device speech-to-text (faster-whisper), semantic intent classification, and voice activity detection (Silero VAD) to deliver a hands-free presenting experience — no cloud APIs, no microphone button mashing, no latency.

**Why?** Traditional presentation tools force speakers to click, tap, or use a clicker. VoiceSlide replaces all of that: just speak naturally, and the system navigates to the right slide, highlights keywords in real time, answers your Q&A from speaker notes, and tracks your speaking analytics — all locally and in real time.

**How?** A Flask + Socket.IO backend streams browser microphone audio through a VAD → Whisper → NLP pipeline. The frontend renders slides with [reveal.js](https://revealjs.com/) and reacts to WebSocket events for navigation, highlighting, and analytics.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Voice Navigation** | Say *"next slide"*, *"go to slide 5"*, or even describe content — the system finds and navigates to the right slide using semantic search. |
| 🎯 **Live Keyword Highlighting** | As you speak, matching words on the current slide are highlighted in real time using fuzzy matching. |
| 🧠 **Semantic Content Search** | Uses sentence-transformer embeddings to match spoken phrases to slide content, even when wording differs. |
| 🎤 **Voice Activity Detection** | Silero VAD detects speech boundaries — transcription only fires when you actually pause, eliminating false triggers. |
| 📊 **Speech Analytics Dashboard** | Post-presentation dashboard with filler word tracking, words-per-minute, and VADER sentiment analysis over time. |
| 💬 **Q&A from Speaker Notes** | Ask a question during your talk and the system searches your speaker notes for relevant answers, displayed on the Presenter Panel. |
| 📑 **PPTX Import** | Upload a `.pptx` file and it's automatically converted to VoiceSlide's slide format — no manual JSON editing required. |
| 🖥️ **Presenter Panel** | A private second-screen view with speaker notes, Q&A results, and current slide context. |
| 🌐 **Fully Local** | All NLP runs on-device. No cloud APIs, no data leaves your machine. |

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | Required for type hints and library compatibility |
| **PyTorch** | 2.x | With CUDA 12.1 for GPU acceleration (CPU works but is slower) |
| **torchaudio** | 2.x | Required by Silero VAD |
| **Node.js** | — | **Not required** — frontend uses CDN-loaded libraries |
| **FFmpeg** | — | **Not required** — faster-whisper handles raw PCM directly |

> [!NOTE]
> PyTorch and torchaudio must be installed **manually** for your CUDA version before running `pip install`. See [Installation](#-installation) below.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone [INSERT_REPO_URL].git
cd voiceslide
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install PyTorch (GPU)

Install PyTorch and torchaudio for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> [!TIP]
> For CPU-only: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

### 4. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

### 5. Add Slide Content

Either upload a `.pptx` through the web UI or place a `slides.json` file in the `data/` directory:

```bash
mkdir -p data
# Option A: Start the server and use /upload in the browser
# Option B: Create data/slides.json manually (see Usage below)
```

### 6. Start the Server

```bash
python backend/app.py
```

The server starts on **`http://localhost:5000`** by default.

---

## 🎮 Usage

### Presenting with Voice Control

1. Open **`http://localhost:5000`** in your browser.
2. Click the **microphone button** (bottom-right) to enable voice input.
3. Speak naturally — VoiceSlide handles the rest:

```
"Next slide"           → advances one slide
"Previous slide"       → goes back one slide
"Go to slide 3"       → jumps to slide 3
"Show the revenue chart" → semantic search finds the matching slide
"First slide"          → jumps to the beginning
"Last slide"           → jumps to the end
```

### Presenter Panel

Open **`http://localhost:5000/presenter`** in a second browser window (or second monitor) to see:
- Current slide speaker notes
- Real-time Q&A results from your notes
- Transcript feed

### Speech Analytics

After your presentation, open **`http://localhost:5000/analytics`** to review:
- **Average WPM** — were you rushing or dragging?
- **Filler word breakdown** — how many "um"s, "like"s, "you know"s?
- **Sentiment timeline** — was your language confident and positive?

### Slide Data Format

VoiceSlide uses a simple JSON format. You can create slides manually or import a `.pptx`:

```json
{
  "slides": [
    {
      "title": "Welcome",
      "content": "<h1>Welcome to VoiceSlide</h1><p>Hands-free presentations.</p>",
      "notes": "Introduce the product and greet the audience."
    },
    {
      "title": "Revenue Growth",
      "content": "<h2>Revenue</h2><p>Revenue grew 18% year over year.</p>",
      "notes": "Key talking point: 18% growth driven by enterprise segment."
    }
  ]
}
```

---

## ⚙️ Configuration

All configuration is managed through environment variables. Defaults are defined in `backend/config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICESLIDE_HOST` | `0.0.0.0` | Server bind address |
| `VOICESLIDE_PORT` | `5000` | Server port |
| `VOICESLIDE_DEBUG` | `true` | Enable Flask debug mode |

```bash
# Example: Run on port 8080 with debug off
export VOICESLIDE_HOST="127.0.0.1"
export VOICESLIDE_PORT="8080"
export VOICESLIDE_DEBUG="false"
python backend/app.py
```

---

## 📁 Project Structure

```
voiceslide/
├── backend/
│   ├── app.py                 # Flask + Socket.IO entry point
│   ├── config.py              # Environment-based configuration
│   ├── transcriber.py         # faster-whisper STT engine
│   ├── vad_engine.py          # Silero VAD speech detection
│   ├── intent_classifier.py   # Voice command classification
│   ├── context_search.py      # Semantic slide search (sentence-transformers)
│   ├── keyword_highlighter.py # Fuzzy keyword matching for live highlights
│   ├── qa_assistant.py        # Q&A from speaker notes
│   ├── analytics.py           # Speech analytics (fillers, WPM, sentiment)
│   ├── slide_loader.py        # JSON slide loading & validation
│   ├── pptx_converter.py      # .pptx → slides.json converter
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html             # Main presentation view
│   ├── upload.html            # Slide upload / editor page
│   ├── presenter.html         # Presenter Panel (speaker notes + Q&A)
│   ├── analytics.html         # Speech Analytics Dashboard
│   ├── css/
│   │   ├── style.css          # Global styles & design tokens
│   │   ├── presentation.css   # Slide presentation styles
│   │   ├── presenter.css      # Presenter Panel styles
│   │   ├── upload.css         # Upload page styles
│   │   └── analytics.css      # Analytics Dashboard styles
│   └── js/
│       ├── app.js             # Main presentation logic + WebSocket
│       ├── presenter.js       # Presenter Panel logic
│       ├── upload.js          # Upload page logic
│       ├── analytics.js       # Analytics Dashboard charts (Chart.js)
│       └── audio-processor.js # AudioWorklet for mic capture
├── tests/
│   ├── test_analytics.py
│   ├── test_context_search.py
│   ├── test_intent_classifier.py
│   ├── test_interceptor.py
│   ├── test_keyword_highlighter.py
│   ├── test_qa_assistant.py
│   ├── test_transcriber.py
│   ├── test_universal_fallback.py
│   └── fixtures/
│       ├── sample.pptx
│       └── sample_audio.raw
└── data/
    └── slides.json            # Active slide content (auto-generated)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow the existing code style — no comments unless they match existing patterns or explain complex logic.
- All new backend features must include tests in `tests/`.
- Run the full test suite before submitting:

```bash
pytest tests/ -v
```

> [!IMPORTANT]
> PyTorch and torchaudio are installed manually and are **not** listed in `requirements.txt`. Make sure your environment has them installed before running tests.

---

## 📄 License

Distributed under the **[INSERT_LICENSE]** License. See `LICENSE` for more information.

---

## 💬 Contact & Acknowledgements

**[Muhammad Hassan]** — [muhammadhassan1762005@gmail.com] — [@MY_LinkedIn](https://www.linkedin.com/in/muhammad-hassan-a3396b290/])

Project Link: [INSERT_REPO_URL]

### Built With

- [Flask](https://flask.palletsprojects.com/) — lightweight Python web framework
- [Socket.IO](https://socket.io/) — real-time bidirectional communication
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [sentence-transformers](https://www.sbert.net/) — semantic text embeddings
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — lexicon-based sentiment analysis
- [reveal.js](https://revealjs.com/) — HTML presentation framework
- [Chart.js](https://www.chartjs.org/) — JavaScript charting library
- [thefuzz](https://github.com/seatgeek/thefuzz) — fuzzy string matching

---

<p align="center">
  Made with ❤️ and a whole lot of voice commands.
</p>
