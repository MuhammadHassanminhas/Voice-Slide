/* ══════════════════════════════════════════════════════════════════════════
   VoiceSlide — Audio Worklet Processor
   Captures raw PCM float32 audio at 16kHz and sends chunks to the main thread.
   ══════════════════════════════════════════════════════════════════════════ */

class VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Send 4000 samples at a time (250ms at 16kHz) to the main thread.
        this.bufferSize = 4000;
        this.buffer = new Float32Array(this.bufferSize);
        this.bytesWritten = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const channelData = input[0];
            for (let i = 0; i < channelData.length; i++) {
                this.buffer[this.bytesWritten++] = channelData[i];

                if (this.bytesWritten >= this.bufferSize) {
                    // Send a copy to the main thread
                    this.port.postMessage(this.buffer.slice(0));
                    this.bytesWritten = 0;
                }
            }
        }
        // Return true to keep the processor alive
        return true;
    }
}

registerProcessor("voice-capture-processor", VoiceCaptureProcessor);
