"use strict";

/* ── DOM References ────────────────────────────────────────────────────── */
const toastContainer = document.getElementById("toast-container");

// Tabs
const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

// PPTX
const dropzone = document.getElementById("upload-dropzone");
const fileInput = document.getElementById("file-input");
const progressContainer = document.getElementById("upload-progress");

// JSON
const jsonInput = document.getElementById("json-input");
const btnSaveJson = document.getElementById("btn-save-json");

// Editor
const slideList = document.getElementById("editor-slide-list");
const formContainer = document.getElementById("editor-form-container");
const btnAddSlide = document.getElementById("btn-add-slide");
const btnSaveEditor = document.getElementById("btn-save-editor");

/* ── State ─────────────────────────────────────────────────────────────── */
let editorSlides = [
    { id: 1, type: "title", heading: "New Presentation", subheading: "", notes: "" }
];
let activeSlideIndex = 0;

/* ── Helpers ───────────────────────────────────────────────────────────── */
function showToast(message, isError = false) {
    const el = document.createElement("div");
    el.className = "toast";
    if (isError) el.style.borderLeftColor = "var(--accent-danger)";
    el.textContent = message;
    toastContainer.appendChild(el);
    setTimeout(() => {
        el.classList.add("toast--exit");
        el.addEventListener("animationend", () => el.remove());
    }, 4000);
}

/* ── Tab Logic ─────────────────────────────────────────────────────────── */
tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        tabPanels.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const target = document.getElementById(btn.dataset.target);
        target.classList.add("active");
    });
});

/* ── 1. PPTX Upload ────────────────────────────────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        handleFileUpload(fileInput.files[0]);
    }
});

async function handleFileUpload(file) {
    if (!file.name.endsWith(".pptx")) {
        showToast("Please upload a valid .pptx file", true);
        return;
    }

    dropzone.style.display = "none";
    progressContainer.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload-pptx", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Upload failed");

        showToast(`✅ Converted ${data.slides_count} slides! Redirecting...`);
        setTimeout(() => { window.location.href = "/"; }, 1500);

    } catch (err) {
        showToast(`❌ Error: ${err.message}`, true);
        dropzone.style.display = "block";
        progressContainer.style.display = "none";
    }
}

/* ── 2. Paste JSON ─────────────────────────────────────────────────────── */
btnSaveJson.addEventListener("click", async () => {
    const code = jsonInput.value.trim();
    if (!code) {
        showToast("JSON cannot be empty", true);
        return;
    }

    try {
        // Basic validation before sending
        const parsed = JSON.parse(code);

        const res = await fetch("/api/save-slides", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(parsed)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Save failed");

        showToast("✅ JSON saved successfully! Redirecting...");
        setTimeout(() => { window.location.href = "/"; }, 1500);

    } catch (err) {
        showToast(`❌ Invalid JSON: ${err.message}`, true);
    }
});

/* ── 3. Web Editor ─────────────────────────────────────────────────────── */
function renderSidebar() {
    slideList.innerHTML = "";
    const icons = {
        title: "T", bullets: "•", image: "🖼️", text: "📄", "two-column": "◫", quote: "❞"
    };

    editorSlides.forEach((slide, index) => {
        const li = document.createElement("li");
        li.className = `slide-item ${index === activeSlideIndex ? "active" : ""}`;

        li.innerHTML = `
      <span class="slide-item__icon">${icons[slide.type] || "📄"}</span>
      <span class="slide-item__title">${slide.heading || `Slide ${index + 1}`}</span>
    `;

        li.addEventListener("click", () => {
            activeSlideIndex = index;
            renderSidebar();
            renderForm();
        });

        slideList.appendChild(li);
    });
}

function renderForm() {
    if (editorSlides.length === 0) {
        formContainer.innerHTML = `<div class="empty-state">No slides. Click + to add one.</div>`;
        return;
    }

    const slide = editorSlides[activeSlideIndex];

    formContainer.innerHTML = `
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Slide Type</label>
        <select class="form-select" id="field-type">
          <option value="title" ${slide.type === 'title' ? 'selected' : ''}>Title Slide</option>
          <option value="bullets" ${slide.type === 'bullets' ? 'selected' : ''}>Bullet Points</option>
          <option value="text" ${slide.type === 'text' ? 'selected' : ''}>Paragraph Text</option>
          <option value="quote" ${slide.type === 'quote' ? 'selected' : ''}>Quote</option>
        </select>
      </div>
    </div>
    
    ${slide.type === 'quote' ? `
      <div class="form-group">
        <label class="form-label">Quote Text</label>
        <textarea class="form-textarea" id="field-quote">${slide.quote || ""}</textarea>
      </div>
      <div class="form-group">
        <label class="form-label">Attribution</label>
        <input type="text" class="form-input" id="field-attribution" value="${slide.attribution || ""}">
      </div>
    ` : `
      <div class="form-group">
        <label class="form-label">Heading</label>
        <input type="text" class="form-input" id="field-heading" value="${slide.heading || ""}">
      </div>
    `}
    
    ${slide.type === 'title' ? `
      <div class="form-group">
        <label class="form-label">Subheading</label>
        <input type="text" class="form-input" id="field-subheading" value="${slide.subheading || ""}">
      </div>
    ` : ''}

    ${slide.type === 'bullets' ? `
      <div class="form-group">
        <label class="form-label">Bullet Items (one per line)</label>
        <textarea class="form-textarea" id="field-items" rows="5">${(slide.items || []).join('\n')}</textarea>
      </div>
    ` : ''}
    
    ${slide.type === 'text' ? `
      <div class="form-group">
        <label class="form-label">Body Text</label>
        <textarea class="form-textarea" id="field-body" rows="6">${slide.body || ""}</textarea>
      </div>
    ` : ''}

    <div class="form-group" style="margin-top:40px; border-top:1px solid rgba(255,255,255,0.1); padding-top:20px;">
      <label class="form-label">Speaker Notes (used for Q&A)</label>
      <textarea class="form-textarea" id="field-notes" rows="3">${slide.notes || ""}</textarea>
    </div>
  `;

}

// Attach fast listener to update model on input (delegated)
formContainer.addEventListener('input', (e) => {
    if (editorSlides.length === 0) return;
    const slide = editorSlides[activeSlideIndex];

    const id = e.target.id;
    const val = e.target.value;

    if (id === 'field-type') {
        slide.type = val;
        renderSidebar();
        renderForm();
    } else if (id === 'field-heading') {
        slide.heading = val;
        if (document.querySelector('.slide-item.active .slide-item__title')) {
            document.querySelector('.slide-item.active .slide-item__title').textContent = val || `Slide ${activeSlideIndex + 1}`;
        }
    } else if (id === 'field-subheading') { slide.subheading = val; }
    else if (id === 'field-quote') { slide.quote = val; }
    else if (id === 'field-attribution') { slide.attribution = val; }
    else if (id === 'field-body') { slide.body = val; }
    else if (id === 'field-items') { slide.items = val.split('\n').filter(i => i.trim()); }
    else if (id === 'field-notes') { slide.notes = val; }
});


btnAddSlide.addEventListener("click", () => {
    editorSlides.push({
        id: editorSlides.length + 1,
        type: "bullets",
        heading: "New Slide",
        items: [],
        notes: ""
    });
    activeSlideIndex = editorSlides.length - 1;
    renderSidebar();
    renderForm();
});

btnSaveEditor.addEventListener("click", async () => {
    const payload = {
        version: "1.0",
        title: editorSlides[0]?.heading || "Editor Presentation",
        author: "Web Editor",
        theme: "dark",
        created_at: new Date().toISOString(),
        slides: editorSlides
    };

    try {
        const res = await fetch("/api/save-slides", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Save failed");

        showToast("✅ Slides saved successfully! Redirecting...");
        setTimeout(() => { window.location.href = "/"; }, 1500);

    } catch (err) {
        showToast(`❌ Error: ${err.message}`, true);
    }
});

// Init editor on load
renderSidebar();
renderForm();
