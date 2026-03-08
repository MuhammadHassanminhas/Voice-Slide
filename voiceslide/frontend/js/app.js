/* ══════════════════════════════════════════════════════════════════════════
   VoiceSlide — Dynamic Slide Renderer  (frontend/js/app.js)
   Fetches slide data from /api/slides, builds reveal.js <section> elements,
   and initializes the presentation.  No hardcoded slide content.
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── DOM References ──────────────────────────────────────────────────────
const slidesContainer = document.getElementById("slides-container");
const noSlidesMsg = document.getElementById("no-slides-message");
const toastContainer = document.getElementById("toast-container");
const micToggleBtn = document.getElementById("mic-toggle-btn");
const transcriptBar = document.getElementById("transcript-bar");
const transcriptText = document.getElementById("transcript-text");

// ── Globals & State ─────────────────────────────────────────────────────
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let audioWorkletNode = null;

// Initialize Socket.IO connection
const socket = io();


// ── Toast Helper ────────────────────────────────────────────────────────
function showToast(message, durationMs = 3000) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  toastContainer.appendChild(el);

  setTimeout(() => {
    el.classList.add("toast--exit");
    el.addEventListener("animationend", () => el.remove());
  }, durationMs);
}

// ── Slide Builders ──────────────────────────────────────────────────────
// Each builder receives a slide object and returns a <section> element.

function buildTitleSlide(slide) {
  const section = document.createElement("section");
  section.className = "slide--title";

  const h1 = document.createElement("h1");
  h1.className = "slide-heading";
  h1.textContent = slide.heading || "";
  section.appendChild(h1);

  if (slide.subheading) {
    const sub = document.createElement("p");
    sub.className = "slide-subheading";
    sub.textContent = slide.subheading;
    section.appendChild(sub);
  }

  return section;
}

function buildBulletsSlide(slide) {
  const section = document.createElement("section");

  const h2 = document.createElement("h2");
  h2.className = "slide-heading";
  h2.textContent = slide.heading || "";
  section.appendChild(h2);

  const ul = document.createElement("ul");
  ul.className = "slide-bullets";
  (slide.items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);
  });
  section.appendChild(ul);

  return section;
}

function buildImageSlide(slide) {
  const section = document.createElement("section");

  const h2 = document.createElement("h2");
  h2.className = "slide-heading";
  h2.textContent = slide.heading || "";
  section.appendChild(h2);

  const wrapper = document.createElement("div");
  wrapper.className = "slide-image-wrapper";

  // Try real image; fall back to styled placeholder
  const img = document.createElement("img");
  img.className = "slide-image";
  img.src = slide.image_url || "";
  img.alt = slide.caption || slide.heading || "Slide image";
  img.onerror = function () {
    // Replace broken image with a nice placeholder
    const ph = document.createElement("div");
    ph.className = "slide-image-placeholder";
    ph.innerHTML = `<span>📊 ${slide.caption || "Image not available"}</span>`;
    wrapper.replaceChild(ph, img);
  };
  wrapper.appendChild(img);

  if (slide.caption) {
    const cap = document.createElement("p");
    cap.className = "slide-image-caption";
    cap.textContent = slide.caption;
    wrapper.appendChild(cap);
  }

  section.appendChild(wrapper);
  return section;
}

function buildTextSlide(slide) {
  const section = document.createElement("section");

  const h2 = document.createElement("h2");
  h2.className = "slide-heading";
  h2.textContent = slide.heading || "";
  section.appendChild(h2);

  const body = document.createElement("div");
  body.className = "slide-body";
  body.textContent = slide.body || "";
  section.appendChild(body);

  return section;
}

function buildTwoColumnSlide(slide) {
  const section = document.createElement("section");

  const h2 = document.createElement("h2");
  h2.className = "slide-heading";
  h2.textContent = slide.heading || "";
  section.appendChild(h2);

  const grid = document.createElement("div");
  grid.className = "slide-columns";

  ["left", "right"].forEach((side) => {
    const col = slide[side] || {};
    const card = document.createElement("div");
    card.className = "slide-column";

    if (col.title) {
      const title = document.createElement("h3");
      title.className = "slide-column__title";
      title.textContent = col.title;
      card.appendChild(title);
    }

    if (col.items && col.items.length) {
      const ul = document.createElement("ul");
      ul.className = "slide-column__items";
      col.items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        ul.appendChild(li);
      });
      card.appendChild(ul);
    }

    grid.appendChild(card);
  });

  section.appendChild(grid);
  return section;
}

function buildQuoteSlide(slide) {
  const section = document.createElement("section");
  section.className = "slide--quote";

  if (slide.heading) {
    const h2 = document.createElement("h2");
    h2.className = "slide-heading";
    h2.textContent = slide.heading;
    section.appendChild(h2);
  }

  const quote = document.createElement("p");
  quote.className = "slide-quote-text";
  quote.textContent = slide.quote || "";
  section.appendChild(quote);

  if (slide.attribution) {
    const attr = document.createElement("p");
    attr.className = "slide-quote-attribution";
    attr.textContent = slide.attribution;
    section.appendChild(attr);
  }

  return section;
}

// ── Builder Dispatch ────────────────────────────────────────────────────
const SLIDE_BUILDERS = {
  title: buildTitleSlide,
  bullets: buildBulletsSlide,
  image: buildImageSlide,
  text: buildTextSlide,
  "two-column": buildTwoColumnSlide,
  quote: buildQuoteSlide,
};

function buildSlideSection(slide) {
  const builder = SLIDE_BUILDERS[slide.type];
  if (!builder) {
    console.warn(`Unknown slide type: "${slide.type}" — skipping.`);
    return null;
  }
  const section = builder(slide);
  section.dataset.slideId = slide.id;
  if (slide.notes) {
    section.dataset.notes = slide.notes;
  }
  return section;
}

// ── Fetch & Render Pipeline ─────────────────────────────────────────────

async function loadAndRenderSlides() {
  try {
    const res = await fetch("/api/slides");
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    const slides = data.slides || [];

    if (slides.length === 0) {
      showNoSlides();
      return;
    }

    // Clear existing slides
    slidesContainer.innerHTML = "";

    // Build sections
    slides.forEach((slide) => {
      const section = buildSlideSection(slide);
      if (section) {
        slidesContainer.appendChild(section);
      }
    });

    // Hide no-slides fallback
    noSlidesMsg.style.display = "none";

    // Initialize reveal.js
    initializeReveal();

    showToast(`✅ Loaded ${slides.length} slides`);
    console.log(`[VoiceSlide] Rendered ${slides.length} slides from /api/slides`);
  } catch (err) {
    console.error("[VoiceSlide] Failed to load slides:", err);
    showNoSlides();
    showToast("❌ Failed to load slides");
  }
}

function showNoSlides() {
  noSlidesMsg.style.display = "flex";
  document.getElementById("reveal-container").style.display = "none";
}

function initializeReveal() {
  Reveal.initialize({
    hash: true,
    slideNumber: true,
    progress: true,
    center: false,
    transition: "slide",
    backgroundTransition: "fade",
    width: 1280,
    height: 720,
    margin: 0.04,
    keyboard: true,
    overview: true,
  });

  // Emit slide-changed events to keep backend in sync (Phase 6)
  Reveal.on("slidechanged", (event) => {
    const slideIndex = event.indexh;
    console.log(`[VoiceSlide] Slide changed → index ${slideIndex}`);

    // Clear highlights from ALL slides on slide change
    document.querySelectorAll(".vs-highlight").forEach((el) => {
      el.classList.remove("vs-highlight", "vs-emphasis");
    });

    socket.emit("slide_changed", { slide_index: slideIndex });
  });
}

// ── Voice Processing (Phase 3) ──────────────────────────────────────────

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    // 16kHz is strictly required for faster-whisper
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,
        channelCount: 1,
      },
    });

    audioContext = new AudioContext({ sampleRate: 16000 });

    // Load the worklet
    await audioContext.audioWorklet.addModule("/js/audio-processor.js");

    const source = audioContext.createMediaStreamSource(mediaStream);
    audioWorkletNode = new AudioWorkletNode(audioContext, "voice-capture-processor");

    // Listen for chunks from the worklet
    audioWorkletNode.port.onmessage = (event) => {
      if (isRecording) {
        // event.data is a Float32Array. We send the raw buffer over the socket.
        socket.emit("audio_chunk", event.data.buffer);
      }
    };

    source.connect(audioWorkletNode);
    audioWorkletNode.connect(audioContext.destination); // Required for processing to run

    isRecording = true;
    micToggleBtn.classList.add("recording");
    transcriptBar.classList.add("visible");
    transcriptText.textContent = "Listening...";
    showToast("🎤 Mic active. Listening...");

  } catch (err) {
    console.error("[VoiceSlide] Failed to access microphone:", err);
    showToast("❌ Could not access microphone. Check permissions.");
  }
}

function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  micToggleBtn.classList.remove("recording");
  transcriptBar.classList.remove("visible");
  transcriptText.textContent = "";

  if (audioWorkletNode) {
    audioWorkletNode.disconnect();
    audioWorkletNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  showToast("⏹️ Mic stopped.");
}

// ── Transcript Display (Phase 3) ─────────────────────────────────────────

socket.on("transcript", (data) => {
  if (!data || !data.text) return;
  transcriptText.textContent = data.text;
});

// Handle incoming navigation commands from backend (Phase 4)
socket.on("nav_command", (data) => {
  if (!data || !data.action) return;

  console.log(`[VoiceSlide] Received Voice Command: ${data.action}`);

  switch (data.action) {
    case "NEXT_SLIDE":
      Reveal.next();
      showToast("▶ Next Slide (Voice Command)");
      break;
    case "PREV_SLIDE":
      Reveal.prev();
      showToast("◀ Previous Slide (Voice Command)");
      break;
    case "NEXT_POINT":
      Reveal.next();
      showToast("▶ Next Point (Voice Command)");
      break;
    case "PREV_POINT":
      Reveal.prev();
      showToast("◀ Previous Point (Voice Command)");
      break;
    case "GOTO_SLIDE":
      if (data.slide_number != null) {
        Reveal.slide(data.slide_number - 1);
        showToast(`⏩ Jump to Slide ${data.slide_number} (Voice Command)`);
      }
      break;
    case "GOTO_CONTENT":
      if (data.slide_number != null) {
        Reveal.slide(data.slide_number - 1);
        showToast(`🔍 Found: Slide ${data.slide_number} (Voice Command)`);
      }
      break;
    case "START_PRESENTATION":
      Reveal.slide(0);
      showToast("🎬 Starting Presentation");
      break;
    case "END_PRESENTATION":
      // Go to the last slide
      const totalSlides = Reveal.getTotalSlides();
      Reveal.slide(totalSlides - 1);
      showToast("🛑 Ending Presentation");
      break;
    default:
      console.warn(`[VoiceSlide] Unknown nav command: ${data.action}`);
  }
});

// ── Phase 6: Live Keyword Highlighting ───────────────────────────────────

function clearHighlights(section) {
  section.querySelectorAll(".vs-highlight").forEach((el) => {
    el.classList.remove("vs-highlight", "vs-emphasis");
  });
}

function highlightSpan(section, spanText, isEmphasis) {
  if (!spanText) return;

  const lowerSpan = spanText.toLowerCase();

  const elements = section.querySelectorAll(
    ".slide-heading, .slide-subheading, .slide-bullets li, " +
    ".slide-body, .slide-image-caption, .slide-quote-text, " +
    ".slide-quote-attribution, .slide-column__title, .slide-column__items li"
  );

  for (const el of elements) {
    if (el.textContent.toLowerCase().includes(lowerSpan)) {
      el.classList.add("vs-highlight");
      if (isEmphasis) {
        el.classList.add("vs-emphasis");
      }

      setTimeout(() => {
        el.classList.remove("vs-highlight", "vs-emphasis");
      }, 3000);

      break;
    }
  }
}

socket.on("highlight_text", (data) => {
  if (!data || data.slide_index == null) return;

  const currentIndex = Reveal.getState().indexh;

  if (data.slide_index !== currentIndex) return;

  const section = document.querySelectorAll(".reveal .slides > section")[currentIndex];
  if (!section) return;

  clearHighlights(section);
  highlightSpan(section, data.matched_span, data.emphasis);
});

// ── Boot ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadAndRenderSlides();
  if (micToggleBtn) {
    micToggleBtn.addEventListener("click", toggleRecording);
  }
});
