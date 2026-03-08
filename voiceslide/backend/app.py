import eventlet
eventlet.monkey_patch()

"""
VoiceSlide — Flask Application (Main Entry Point)
Serves the frontend, exposes REST API for slide data, and
will later host WebSocket events for real-time features.
"""

import logging
import sys
import os

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure backend/ is on sys.path so modules can be imported cleanly
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO

import config
from slide_loader import load_slides, save_slides
from pptx_converter import convert_pptx
import uuid
from werkzeug.utils import secure_filename

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voiceslide")

# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    static_folder=config.FRONTEND_DIR,
    static_url_path="",
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


# ── Startup: Index slide content for context-aware navigation (Phase 5) ──────
import context_search
import qa_assistant
import analytics

def _index_current_slides():
    """Build the context search index from the current slides.json."""
    try:
        data = load_slides(config.SLIDES_JSON_PATH)
        context_search.build_index(data.get("slides", []))
        qa_assistant.build_notes_index()
    except FileNotFoundError:
        logger.info("No slides.json found — context search index empty")
    except Exception as exc:
        logger.warning("Failed to build context index: %s", exc)

_index_current_slides()


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main presentation page."""
    return send_from_directory(config.FRONTEND_DIR, "index.html")

@app.route("/upload")
def upload_page():
    """Serve the upload/editor page."""
    return send_from_directory(config.FRONTEND_DIR, "upload.html")

@app.route("/presenter")
def presenter_page():
    """Serve the private Presenter Panel page."""
    return send_from_directory(config.FRONTEND_DIR, "presenter.html")

@app.route("/analytics")
def analytics_page():
    """Serve the Speech Analytics Dashboard page."""
    return send_from_directory(config.FRONTEND_DIR, "analytics.html")


# Static images served from frontend/static/images/
@app.route("/static/images/<path:filename>")
def serve_image(filename):
    """Serve images from the static images directory."""
    images_dir = os.path.join(config.STATIC_DIR, "images")
    return send_from_directory(images_dir, filename)


# Explicit CSS / JS routes (belt-and-suspenders for eventlet compatibility)
@app.route("/css/<path:filename>")
def serve_css(filename):
    """Serve CSS files from frontend/css/."""
    return send_from_directory(os.path.join(config.FRONTEND_DIR, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve JavaScript files from frontend/js/."""
    return send_from_directory(os.path.join(config.FRONTEND_DIR, "js"), filename)


# ── API Routes ───────────────────────────────────────────────────────────────

@app.route("/api/analytics", methods=["GET"])
def api_get_analytics():
    """Return the current session's speech analytics as JSON."""
    return jsonify(analytics.tracker.get_summary()), 200

@app.route("/api/slides", methods=["GET"])
def api_get_slides():
    """Return the current slide data as JSON.

    Returns:
        200 — full slides.json content
        404 — if slides.json does not exist
        500 — if file is invalid
    """
    try:
        data = load_slides(config.SLIDES_JSON_PATH)
        logger.info("GET /api/slides — served %d slides", len(data.get("slides", [])))
        return jsonify(data), 200
    except FileNotFoundError:
        logger.warning("GET /api/slides — slides.json not found")
        return jsonify({"error": "No slides found. Upload a presentation or create one."}), 404
    except ValueError as exc:
        logger.error("GET /api/slides — validation error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/api/upload-pptx", methods=["POST"])
def api_upload_pptx():
    """Accept a .pptx file, convert it to slides.json, and save it."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith(".pptx"):
        return jsonify({"error": "Invalid file type. Only .pptx is supported."}), 400
        
    try:
        filename = secure_filename(file.filename)
        temp_filename = f"{uuid.uuid4().hex}_{filename}"
        temp_path = os.path.join(config.UPLOAD_DIR, temp_filename)
        file.save(temp_path)
        
        logger.info("Parsing uploaded PPTX: %s", temp_filename)
        slide_data = convert_pptx(temp_path)
        
        save_slides(slide_data, config.SLIDES_JSON_PATH)
        context_search.build_index(slide_data.get("slides", []))
        qa_assistant.build_notes_index()

        try:
            os.remove(temp_path)
        except OSError:
            pass
            
        return jsonify({
            "message": "Successfully uploaded and converted", 
            "slides_count": len(slide_data.get("slides", []))
        }), 200
        
    except Exception as exc:
        logger.error("Error processing PPTX: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/api/save-slides", methods=["POST"])
def api_save_slides():
    """Accept raw JSON slide data, validate it, and save it."""
    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "No JSON payload provided"}), 400
             
        save_slides(data, config.SLIDES_JSON_PATH)
        context_search.build_index(data.get("slides", []))
        qa_assistant.build_notes_index()
        return jsonify({"message": "Successfully saved slides"}), 200
    except ValueError as exc:
        logger.error("Validation error when saving slides: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Unexpected error saving slides: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ── WebSocket Events (Phase 4.1 — VAD-driven pipeline) ───────────────────────

from transcriber import transcribe_chunk
from intent_classifier import classify_intent
from vad_engine import get_vad_engine
from keyword_highlighter import fuzzy_match_current_slide
import time

# Cooldown state for Voice Navigation
_last_nav_command_time = 0
NAV_COMMAND_COOLDOWN = 1.0  # seconds

# Phase 6: Track which slide the frontend is currently showing
_current_slide_index = 0

# VAD-driven speech accumulation (replaces the Phase 3 rolling buffer)
_speech_buffer = bytearray()
_silence_counter = 0
_is_speaking = False

SILENCE_CHUNKS_THRESHOLD = 1        # 1 × 250 ms = 250 ms of post-speech silence
MAX_SPEECH_DURATION_BYTES = 480000   # 30 s safety cap  (16000 Hz × 4 B × 30 s)


@socketio.on("connect")
def handle_connect():
    logger.info("WebSocket client connected")


@socketio.on("disconnect")
def handle_disconnect():
    global _silence_counter, _is_speaking, _current_slide_index
    logger.info("WebSocket client disconnected")
    _speech_buffer.clear()
    _silence_counter = 0
    _is_speaking = False
    _current_slide_index = 0


@socketio.on("slide_changed")
def handle_slide_changed(data):
    global _current_slide_index
    idx = data.get("slide_index", 0) if isinstance(data, dict) else 0
    _current_slide_index = int(idx)
    logger.debug("Slide index updated → %d", _current_slide_index)


@socketio.on("reset_analytics")
def handle_reset_analytics():
    """Reset analytics tracker for a new presentation session."""
    analytics.tracker.reset()
    logger.info("Analytics session reset.")
    socketio.emit("analytics_reset")


def _update_current_index(payload):
    """Best-effort pre-sync of _current_slide_index after emitting a nav_command."""
    global _current_slide_index
    action = payload.get("action")
    if action in ("NEXT_SLIDE", "NEXT_POINT"):
        _current_slide_index = min(_current_slide_index + 1, max(context_search._num_slides - 1, 0))
    elif action in ("PREV_SLIDE", "PREV_POINT"):
        _current_slide_index = max(_current_slide_index - 1, 0)
    elif action in ("GOTO_SLIDE", "GOTO_CONTENT"):
        sn = payload.get("slide_number")
        if sn is not None:
            _current_slide_index = sn - 1
    elif action == "START_PRESENTATION":
        _current_slide_index = 0
    elif action == "END_PRESENTATION":
        _current_slide_index = max(context_search._num_slides - 1, 0)


def _process_speech_buffer():
    """Transcribe the accumulated speech buffer, check for nav intents, then reset state."""
    global _last_nav_command_time, _silence_counter, _is_speaking

    # Snapshot the buffer and reset state immediately so the next utterance
    # starts clean even if transcription is slow.
    buffer_snapshot = bytes(_speech_buffer)
    _speech_buffer.clear()
    _silence_counter = 0
    _is_speaking = False
    get_vad_engine().reset()

    if len(buffer_snapshot) == 0:
        return

    # 1. Transcribe the complete utterance
    result = transcribe_chunk(buffer_snapshot)
    transcribed_text = result.get("text", "")

    if not transcribed_text:
        return

    socketio.emit("transcript", result)

    # ── Phase 9: Record speech analytics ─────────────────────────
    analytics.tracker.record_segment(transcribed_text)

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

    # 2. Classify intent
    intent_result = classify_intent(transcribed_text)
    command = intent_result.get("intent")

    if command and command != "NONE":
        current_time = time.time()
        if current_time - _last_nav_command_time > NAV_COMMAND_COOLDOWN:
            _last_nav_command_time = current_time
            logger.info(
                "Emitting nav_command: %s (Confidence: %.2f)",
                command, intent_result.get("confidence", 0.0),
            )

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
            _update_current_index(payload)
        else:
            logger.debug("Ignored nav_command %s due to cooldown.", command)
    else:
        # ── Phase 6 Interceptor — Fuzzy Match against CURRENT slide ───
        highlight_result = fuzzy_match_current_slide(transcribed_text, _current_slide_index)

        if highlight_result is not None:
            socketio.emit("highlight_text", highlight_result)
            logger.debug(
                "Intercepted: '%s' matches current slide %d (score: %d)",
                transcribed_text, _current_slide_index, highlight_result["score"],
            )
            return

        # ── Universal Fallback — classifier returned NONE ────────────
        match = context_search.search(transcribed_text)
        if match:
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
                _update_current_index({
                    "action": "GOTO_CONTENT",
                    "slide_number": match["slide_index"] + 1,
                })
            else:
                logger.debug("Ignored fallback GOTO_CONTENT due to cooldown.")
        else:
            logger.debug("No intent or content match for: '%s'", transcribed_text)


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Receive a 250 ms binary audio chunk (raw PCM float32, 16 kHz, mono).
    Uses Silero VAD to detect speech boundaries and only fires transcription
    when the speaker actually pauses.
    """
    global _is_speaking, _silence_counter

    if not isinstance(data, bytes) or len(data) == 0:
        return

    vad = get_vad_engine()
    confidence = vad.get_speech_confidence(data)
    logger.debug("VAD confidence: %.2f", confidence)

    if confidence >= vad.threshold:
        # ── Speech detected ──────────────────────────────────────────
        _is_speaking = True
        _silence_counter = 0
        _speech_buffer.extend(data)

    elif _is_speaking:
        # ── Silence after speech — pad buffer and count ──────────────
        _speech_buffer.extend(data)
        _silence_counter += 1

        if _silence_counter >= SILENCE_CHUNKS_THRESHOLD:
            logger.info(
                "Speech End detected, transcribing... (%d bytes)",
                len(_speech_buffer),
            )
            _process_speech_buffer()

    # else: pure silence while not speaking → discard chunk

    # ── Safety cap: force-transcribe if buffer grows too large ────────
    if len(_speech_buffer) > MAX_SPEECH_DURATION_BYTES:
        logger.warning(
            "Speech buffer exceeded max duration (%d bytes), force-transcribing.",
            len(_speech_buffer),
        )
        _process_speech_buffer()

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting VoiceSlide server on http://localhost:%d", config.PORT)
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG)
