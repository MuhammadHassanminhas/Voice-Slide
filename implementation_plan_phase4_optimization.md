# Implementation Plan - Phase 4 Optimization (Smart VAD & NLP Heuristics)

We will optimize the Voice Navigation system to eliminate false positives (mid-sentence triggering) and false negatives (chopped commands) by replacing the timer-based buffer with Semantic Voice Activity Detection (VAD) and improving the Intent Classifier with linguistic rules.

## 1. Dependencies & Setup

We need `silero-vad` for robust speech detection. It runs on the existing PyTorch installation.

-   **Modify `backend/requirements.txt`**:
    -   Add `torchaudio` (required by Silero VAD).
    -   Add `omegaconf` (dependency for Silero VAD loading).

## 2. Backend: Voice Activity Detection (VAD) Engine

We will create a dedicated VAD module to manage the speech states.

-   **Create `backend/vad_engine.py`**:
    -   **Class `VADEngine`**:
        -   **`__init__`**: Load the `silero_vad` model from `torch.hub` (lazy loaded singleton).
        -   **`is_speech(audio_chunk)`**: Accepts a binary audio chunk (float32), converts it to a Tensor, and runs the VAD model. Returns a confidence score (0.0 to 1.0).
    -   **Speech State Machine**:
        -   Instead of just "is speech", we need a logic wrapper in `app.py` or here that tracks:
            -   `SILENCE` (Collecting data, but dropping old data if too long)
            -   `SPEECH_START` (Triggered when confidence > 0.5)
            -   `SPEECH_ONGOING` (Accumulating buffer)
            -   `SPEECH_END` (Triggered when silence persists for > 400ms after speech)

## 3. Backend: Audio Pipeline Refactor (`app.py`)

We will replace the current Rolling Buffer (time-based) with a VAD-driven Event Loop.

-   **Modify `backend/app.py`**:
    -   **Remove**: `MIN_BUFFER_BYTES`, `MAX_BUFFER_BYTES` (rolling logic), and the time-based transcription trigger.
    -   **State Variables**:
        -   `speech_buffer`: Bytearray to accumulate audio *during* a speech utterance.
        -   `silence_counter`: Integers counting consecutive silent chunks to detect "End of Speech".
        -   `is_speaking`: Boolean flag.
    -   **Update `handle_audio_chunk`**:
        1.  Receive 250ms chunk from frontend.
        2.  Run `VADEngine.is_speech(chunk)`.
        3.  **Logic**:
            -   **If Speaking (`conf > 0.5`)**:
                -   Set `is_speaking = True`.
                -   Reset `silence_counter`.
                -   Append chunk to `speech_buffer`.
            -   **If Silence (`conf < 0.5`)**:
                -   If `is_speaking` is True:
                    -   Append chunk to `speech_buffer` (padding).
                    -   Increment `silence_counter`.
                    -   **If `silence_counter` > threshold (e.g. 2 chunks / 500ms)**:
                        -   **FIRE TRANSCRIPTION**: Send `speech_buffer` to `transcriber.py`.
                        -   Emit `transcript`.
                        -   Check `classify_intent`.
                        -   Reset `speech_buffer`, `is_speaking`, `silence_counter`.
                -   If `is_speaking` is False:
                    -   Discard chunk (ignore background noise/silence).

## 4. NLP Heuristics: Intent Classifier Updates

We will make the classifier smarter about *usage context* to prevent "Next slide is important" from triggering navigation.

-   **Modify `backend/intent_classifier.py`**:
    -   **Add Heuristics Logic** inside `classify_intent(text)` before calling the embedder.
    -   **Heuristic 1: Length Penalty**:
        -   If `len(text.split()) > 10`: Return `NONE` (Commands are rarely long sentences).
        -   *Exception*: If the text *starts* with a strong command phrase like "VoiceSlide go to...", but for now, hard length limit is safer for navigation.
    -   **Heuristic 2: End-of-Sentence Bias**:
        -   If a Fast-Path exact match is found (e.g., "next slide"), check its position.
        -   Valid: "Please go to the next slide" (End of sentence).
        -   Valid: "Next slide" (Whole sentence).
        -   Invalid: "The next slide shows our revenue" (Middle of sentence).
        -   **Matches**: stricter RegEx or string checks (`text.lower().endswith(phrase)`).

## 5. Verification Plan

1.  **Setup Check**: Verify `torchaudio` installs and VAD model loads.
2.  **VAD Test**: Speak a sentence, pause. Verify `app.py` logs "Speech End detected, transcribing..." exactly once after the pause.
3.  **Noise Test**: Remain silent. Verify no transcripts are generated (silence suppression).
4.  **Intent Test (False Positive)**: Say "The PREVIOUS SLIDE was boring." -> Should NOT trigger.
5.  **Intent Test (True Positive)**: Say "Go to the previous slide." -> Should trigger.

## 6. Execution Order

1.  Update `requirements.txt`.
2.  Create `backend/vad_engine.py`.
3.  Refactor `backend/app.py` (Audio Chunk Loop).
4.  Refactor `backend/intent_classifier.py` (Heuristics).
