"""
VoiceSlide — Voice Activity Detection (VAD) Engine
Uses Silero VAD to determine whether an audio chunk contains speech.
Singleton pattern — one model instance shared across all requests.

Audio contract:
    Input  — raw bytes, PCM float32, 16 kHz, mono  (same as transcriber.py)
    Output — confidence float [0.0, 1.0]
"""

import logging

import numpy as np
import torch

logger = logging.getLogger("voiceslide.vad")

# ── Constants ────────────────────────────────────────────────────────────────

_SAMPLE_RATE = 16000
_VAD_WINDOW_SAMPLES = 512  # Silero VAD window size for 16 kHz — 7 passes per 4000-sample chunk


# ── VADEngine Class ──────────────────────────────────────────────────────────

class VADEngine:
    """Wraps the Silero VAD model for per-chunk speech confidence scoring."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

        logger.info("Loading Silero VAD model from torch.hub ...")
        try:
            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        except Exception as exc:
            logger.error("Failed to load Silero VAD model: %s", exc)
            raise

        # ── Device selection (CUDA with CPU fallback, mirrors transcriber.py) ──
        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device)
        except Exception as exc:
            logger.warning("CUDA unavailable for VAD, falling back to CPU: %s", exc)
            self._device = "cpu"
            self._model = model

        self._model.reset_states()
        logger.info(
            "Silero VAD model loaded on %s (threshold=%.2f).",
            self._device, threshold,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def get_speech_confidence(self, audio_bytes: bytes) -> float:
        """
        Compute speech confidence for a raw PCM float32 audio chunk.

        Args:
            audio_bytes: Raw binary PCM float32 audio (16 kHz, mono).

        Returns:
            Confidence score between 0.0 (silence) and 1.0 (speech).
        """
        if len(audio_bytes) == 0:
            return 0.0

        try:
            # bytes → numpy float32 → torch tensor
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32).copy()

            if not np.isfinite(audio_array).all():
                logger.warning("Audio chunk contains NaN/Inf values, treating as silence.")
                return 0.0

            tensor = torch.from_numpy(audio_array).to(self._device)

            # Silero VAD requires 512-sample windows at 16 kHz.
            # Transfer the full tensor once, then slice on-device (zero-copy views).
            max_conf = 0.0
            num_samples = len(tensor)
            for start in range(0, num_samples - _VAD_WINDOW_SAMPLES + 1, _VAD_WINDOW_SAMPLES):
                window = tensor[start : start + _VAD_WINDOW_SAMPLES]
                conf = float(self._model(window, _SAMPLE_RATE))
                if conf > max_conf:
                    max_conf = conf

            return max_conf

        except Exception as exc:
            logger.error("VAD inference error: %s", exc)
            return 0.0

    def is_speech(self, audio_bytes: bytes) -> bool:
        """Return True if the audio chunk contains speech above the threshold."""
        return self.get_speech_confidence(audio_bytes) >= self.threshold

    def reset(self) -> None:
        """Reset LSTM hidden states.  Call between utterances."""
        self._model.reset_states()
        # Guard: some Silero versions create hidden-state tensors on CPU
        # inside reset_states() even when the model lives on GPU.
        if self._device != "cpu":
            try:
                for attr in ("_h", "_c", "h", "c"):
                    t = getattr(self._model, attr, None)
                    if isinstance(t, torch.Tensor) and t.device.type != self._device:
                        setattr(self._model, attr, t.to(self._device))
            except Exception:
                pass  # JIT model manages its own device placement


# ── Singleton Access ─────────────────────────────────────────────────────────

_engine: VADEngine | None = None


def get_vad_engine() -> VADEngine:
    """Return the shared VADEngine instance (created on first call)."""
    global _engine
    if _engine is None:
        _engine = VADEngine()
    return _engine
