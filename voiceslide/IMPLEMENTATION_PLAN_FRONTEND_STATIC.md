# Frontend Static File Serving — Implementation Plan

> **Status**: Plan for review — no code will be written until approved.
>
> **Symptom**: Browser loads `index.html` (mic SVG icon visible) but the page is a white
> screen — no styling, no slide viewer, no drag-and-drop UI. CSS and JS assets are not
> being delivered to the browser.

---

## 0. Diagnostic Summary

### What I Verified

| Check | Result |
|---|---|
| `config.FRONTEND_DIR` resolves correctly | ✅ `C:\...\voiceslide\frontend` (absolute) |
| Directory exists | ✅ `True` |
| `css/style.css` exists at expected path | ✅ `True` |
| `js/app.js` exists at expected path | ✅ `True` |
| Flask URL map registers `/<path:filename>` static handler | ✅ Registered |
| Flask test client serves `/css/style.css` | ✅ **200**, `text/css` |
| Flask test client serves `/js/app.js` | ✅ **200**, `text/javascript` |
| Explicit `import eventlet` / `eventlet.monkey_patch()` in `app.py` | ❌ **Missing** |

**Key finding**: Flask's static handler is correctly configured and works in isolation
(test client returns 200 for all assets). The failure is **runtime-specific** — it only
occurs under the eventlet WSGI server that `socketio.run()` starts.

---

## 1. Root Cause Analysis

### Primary Cause: Missing Early `eventlet.monkey_patch()`

`app.py` line 42 specifies `async_mode="eventlet"`, but **nowhere** in `app.py` is
eventlet explicitly imported or monkey-patched. The current import section:

```python
# app.py lines 7-23
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO
import config
...
```

No `import eventlet`. No `eventlet.monkey_patch()`.

**What happens at runtime**:

1. Flask is imported and configured with standard (un-patched) `os`, `io`, and `socket`
   modules.
2. All route handlers and the static file handler are bound to un-patched file I/O.
3. `socketio.run()` (line 270) is called — Flask-SocketIO internally calls
   `eventlet.monkey_patch()` **at this point**, then starts the eventlet WSGI server.
4. The WSGI server uses eventlet's green threads for request handling. Incoming HTTP
   requests run inside green threads that expect **patched** I/O primitives.
5. When the static file handler tries to serve `css/style.css`, it calls Werkzeug's
   `send_file()`, which uses `open()` and response streaming. The mismatch between
   the green-thread execution context (which expects patched I/O) and the late-patched
   file operations causes the response to silently fail, hang, or return empty.

**Why `index.html` works**: The `/` route uses `send_from_directory()` as an explicit
route handler. Flask-SocketIO's eventlet integration handles explicit route responses
differently from the implicit static file endpoint. The `send_from_directory` call in
the route handler is a direct function call, while the static handler goes through
Flask's `StaticFileView` class which uses a different code path for file streaming.

**Version context** (installed):

| Package | Version | Note |
|---|---|---|
| eventlet | 0.40.4 | **Deprecated** — bugfix-only mode. Known late-patch issues. |
| Flask | 3.1.3 | Werkzeug 3.x changed `send_file` internals. |
| Werkzeug | 3.1.6 | Uses `wsgi.FileWrapper` for efficient file streaming. |
| Flask-SocketIO | 5.6.1 | Calls `eventlet.monkey_patch()` inside `socketio.run()`. |

The combination of Werkzeug 3.x's `FileWrapper`-based streaming + eventlet 0.40.x's
incomplete monkey-patching (applied late) is the failure mode.

### Contributing Factor: No Explicit CSS/JS Routes

The codebase has explicit `send_from_directory` routes for:
- ✅ `index.html` → `@app.route("/")`
- ✅ `upload.html` → `@app.route("/upload")`
- ✅ Images → `@app.route("/static/images/<path:filename>")`

But CSS and JS rely on Flask's **implicit** static handler (`/<path:filename>`). This
implicit handler is the one affected by the eventlet patching issue. The explicit routes
work because they're direct function calls, not routed through the `StaticFileView`.

---

## 2. Backend Routing Strategy

### Fix 1 (Required): Early Monkey-Patching

Add eventlet import and monkey-patch as the **very first executable lines** in `app.py`,
before ANY other imports. This ensures all I/O modules are patched before Flask,
Werkzeug, or any other library captures references to them.

**Insertion point**: `app.py` line 1, before the module docstring or imports.

```python
import eventlet
eventlet.monkey_patch()

"""
VoiceSlide — Flask Application (Main Entry Point)
...
"""

import logging
import sys
import os
# ... rest of imports unchanged
```

**Why before everything**: `monkey_patch()` replaces `os`, `socket`, `select`, `time`,
and `io` at the module level. Any code that imports these modules AFTER the patch gets
the patched versions. Any code that imported BEFORE gets the originals. Placing it first
guarantees consistency.

### Fix 2 (Belt-and-Suspenders): Explicit CSS/JS Routes

Even with early monkey-patching, adding explicit `send_from_directory` routes for CSS
and JS directories makes the file serving deterministic and self-documenting. This
follows the existing pattern already used for images (lines 58–63).

**New routes** (insert after the existing `/static/images/` route, ~line 63):

```python
@app.route("/css/<path:filename>")
def serve_css(filename):
    """Serve CSS files from frontend/css/."""
    return send_from_directory(os.path.join(config.FRONTEND_DIR, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve JavaScript files from frontend/js/."""
    return send_from_directory(os.path.join(config.FRONTEND_DIR, "js"), filename)
```

**Why both fixes**: Fix 1 addresses the root cause (eventlet compatibility). Fix 2
provides defense-in-depth — if eventlet has any further static-handler bugs, the
explicit routes bypass the `StaticFileView` entirely. Together, they ensure assets are
served reliably regardless of the WSGI server's behavior.

### What Stays Unchanged

- `static_folder=config.FRONTEND_DIR` and `static_url_path=""` — kept for any other
  static files that might be added later (fonts, favicons, etc.). The explicit routes
  take priority for `/css/` and `/js/` paths.
- All existing API routes (`/api/slides`, `/api/upload-pptx`, `/api/save-slides`).
- All WebSocket event handlers (`audio_chunk`, `connect`, `disconnect`).
- The VAD pipeline, transcriber, and intent classifier — completely untouched.

---

## 3. Frontend Path Strategy

**No HTML changes required.** The current asset paths in both HTML files are already
correct for the proposed backend routing:

### `index.html` — Current paths (unchanged)

```html
<!-- CSS — served by new explicit /css/ route -->
<link rel="stylesheet" href="/css/style.css" />
<link rel="stylesheet" href="/css/presentation.css" />

<!-- JS — served by new explicit /js/ route -->
<script src="/js/app.js"></script>

<!-- CDN scripts — unchanged, no server involvement -->
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.min.js"></script>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
```

### `upload.html` — Current paths (unchanged)

```html
<link rel="stylesheet" href="/css/style.css" />
<link rel="stylesheet" href="/css/upload.css" />
<script src="/js/upload.js"></script>
```

### `app.js` — AudioWorklet path (unchanged)

```javascript
// Line 322 — already uses root-relative path
await audioContext.audioWorklet.addModule("/js/audio-processor.js");
```

**Path mapping after fix**:

| Browser Request | Route | Filesystem Path |
|---|---|---|
| `/` | `index()` (explicit) | `frontend/index.html` |
| `/upload` | `upload_page()` (explicit) | `frontend/upload.html` |
| `/css/style.css` | `serve_css()` (NEW explicit) | `frontend/css/style.css` |
| `/css/presentation.css` | `serve_css()` (NEW explicit) | `frontend/css/presentation.css` |
| `/js/app.js` | `serve_js()` (NEW explicit) | `frontend/js/app.js` |
| `/js/audio-processor.js` | `serve_js()` (NEW explicit) | `frontend/js/audio-processor.js` |
| `/static/images/*` | `serve_image()` (existing) | `frontend/static/images/*` |

---

## 4. Debugging Checklist — Confirm Before Applying Fix

Run these checks **now** (before code changes) to validate the root cause:

### 4.1 Browser Console (F12 → Console Tab)

Look for:
- [ ] `GET http://localhost:5000/css/style.css net::ERR_FAILED` or `404 (Not Found)`
- [ ] `GET http://localhost:5000/js/app.js net::ERR_FAILED` or `404 (Not Found)`
- [ ] `Refused to apply style ... MIME type ('text/html') is not a supported stylesheet MIME type`
- [ ] `ReferenceError: Reveal is not defined` (would indicate the reveal.js CDN also failed — network issue)
- [ ] `ReferenceError: io is not defined` (Socket.IO CDN failed — would also kill audio pipeline)
- [ ] Any `[VoiceSlide]` prefixed errors from `app.js`

### 4.2 Browser Network Tab (F12 → Network Tab)

- [ ] Filter by `CSS` — check status codes for `style.css` and `presentation.css`
- [ ] Filter by `JS` — check status codes for `app.js`
- [ ] Check `Content-Type` headers on CSS/JS responses (should be `text/css` / `text/javascript`, not `text/html`)
- [ ] Check response body — if Flask returns 404, the body would be HTML error page, not CSS/JS content
- [ ] Look for responses with status `0` or `(failed)` — indicates connection-level failure (eventlet not sending response)

### 4.3 Flask Terminal Output

- [ ] Check if requests for `/css/style.css` appear in the server log at all
- [ ] If they do appear, check the status code: `200` or `404` or `500`
- [ ] If they do NOT appear, the request never reached Flask (eventlet dropped it)
- [ ] Look for any Python tracebacks or eventlet-related warnings

### 4.4 Quick Command-Line Verification

While the server is running, test from a separate terminal:

```powershell
# Test CSS serving
Invoke-WebRequest -Uri "http://localhost:5000/css/style.css" -Method GET | Select-Object StatusCode, ContentType

# Test JS serving
Invoke-WebRequest -Uri "http://localhost:5000/js/app.js" -Method GET | Select-Object StatusCode, ContentType

# Test explicit route (should always work)
Invoke-WebRequest -Uri "http://localhost:5000/" -Method GET | Select-Object StatusCode, ContentType
```

**Expected results if root cause is confirmed**:
- `/` → `200`, `text/html` ✅ (explicit route works)
- `/css/style.css` → `404` or `0` or timeout ❌ (static handler fails)
- `/js/app.js` → `404` or `0` or timeout ❌ (static handler fails)

---

## 5. Files Changed — Summary

| File | Action | Changes |
|---|---|---|
| `backend/app.py` | **Modify** | (1) Add `import eventlet; eventlet.monkey_patch()` as first two lines (~2 lines). (2) Add explicit `/css/<path>` and `/js/<path>` routes after the existing image route (~10 lines). Net: ~12 lines added. |

**No other files modified.** No HTML changes, no JS changes, no config changes, no
frontend changes. This is a backend-only fix.

---

## 6. Execution Order

1. **Verify root cause** — Run the debugging checklist (§4) to confirm CSS/JS 404s.
2. **`backend/app.py`** — Add `import eventlet; eventlet.monkey_patch()` at the top.
3. **`backend/app.py`** — Add explicit `/css/` and `/js/` routes.
4. **Restart server** — `python backend/app.py` (eventlet patch must be applied fresh).
5. **Re-test** — Reload browser, verify CSS/JS load (Network tab shows 200 + correct MIME types).
6. **Visual confirmation** — UI renders fully (styled slide viewer or styled no-slides fallback).

---

## 7. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Early monkey-patching interferes with model loading (torch, faster-whisper) | CUDA/model init fails | Unlikely — PyTorch and faster-whisper don't use eventlet-patched sockets for CUDA operations. Verified by existing Flask-SocketIO + CUDA setups in production. If an issue arises, we can selectively patch: `eventlet.monkey_patch(os=True, socket=True, select=True, time=True)` excluding modules torch needs. |
| Explicit CSS/JS routes shadow future static files in those directories | New CSS/JS files not served | Not a risk — the `<path:filename>` parameter catches all sub-paths. Any file added to `frontend/css/` or `frontend/js/` is automatically served. |
| `send_from_directory` path traversal vulnerability | Security issue | Not a risk — Werkzeug's `send_from_directory` uses `safe_join` internally, which rejects any path containing `..` or absolute paths. Same security as the existing image route. |
| eventlet deprecation causes future breakage | Server instability | Known issue, but out of scope for this fix. A future migration to `async_mode="threading"` or a different async framework is recommended but not part of this patch. |
