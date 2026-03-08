"""
NLP Engine — Transcriber Module
Handles speech-to-text using `faster-whisper`.
Designed as a singleton to keep the model loaded in memory for fast audio chunk transcription.
"""

# ── CUDA DLL Preload (Windows only) ─────────────────────────────────────
# CTranslate2 (used by faster-whisper) needs cuBLAS/cuDNN DLLs at runtime.
# PyTorch bundles these in its own `torch/lib/` folder.  On Windows the OS
# won't find them unless we explicitly register that directory BEFORE any
# ctranslate2 / faster_whisper import touches native code.
import os
import sys

if sys.platform == "win32":
    try:
        import torch as _torch
        _cuda_lib_dir = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_cuda_lib_dir):
            os.add_dll_directory(_cuda_lib_dir)
    except ImportError:
        pass  # torch not installed — will fall back to CPU anyway

import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("voiceslide.transcriber")

# ── Singleton Model ──────────────────────────────────────────────────────────
_model = None

def get_model() -> WhisperModel:
    """
    Lazy-load and return the faster-whisper model.
    Attempts CUDA/float16 first, falls back to CPU/int8.
    """
    global _model
    if _model is None:
        logger.info("Loading faster-whisper 'base' model...")
        try:
            _model = WhisperModel("base", device="cuda", compute_type="float16")
            logger.info("Model loaded successfully on CUDA (float16).")
        except Exception as e:
            logger.warning(f"CUDA initialization failed for faster-whisper: {e}. Falling back to CPU (int8).")
            _model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("Model loaded successfully on CPU (int8).")
            
    return _model

# ── Transcription Logic ──────────────────────────────────────────────────────

def transcribe_chunk(audio_bytes: bytes) -> dict:
    """
    Transcribe raw PCM float32 audio bytes into text.
    
    Args:
        audio_bytes: Raw binary PCM float32 audio (16kHz, mono).
        
    Returns:
        dict: {"text": str, "is_final": bool, "language": str}
    """
    try:
        model = get_model()
        
        # Convert raw bytes to a 1D NumPy float32 array
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if len(audio_array) == 0:
            return {"text": "", "is_final": True, "language": "en"}
            
        # Transcribe the audio chunk
        segments, info = model.transcribe(audio_array, language="en", beam_size=1)
        
        text = "".join(segment.text for segment in segments).strip()
        
        return {
            "text": text,
            "is_final": True, # This simple approach treats every chunk as final for now
            "language": info.language
        }
    except Exception as exc:
        logger.error("Error transcribing audio chunk: %s", exc)
        return {"text": "", "is_final": True, "language": "en"}
