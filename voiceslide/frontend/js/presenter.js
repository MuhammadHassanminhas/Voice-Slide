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
