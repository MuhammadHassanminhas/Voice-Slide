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
    loadAnalytics();
});

// ── Initial Load ───────────────────────────────────────────────────────
loadAnalytics();
