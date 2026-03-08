# Phase 5 Semantic Fix — Query Cleaning & Dual-Embedding Index

> **Status**: Blueprint for review — **no code will be written until approved.**
>
> **Symptom**: When the Universal Fallback passes raw transcribed text like
> *"move to the RISC slide"* to `context_search.search()`, the navigational
> stop-words (`move`, `to`, `the`, `slide`) pollute the embedding vector.
> Meanwhile, the single flattened body text per slide (heading + bullets +
> notes all concatenated) dilutes topic-specific keywords in the embedding
> space, making title-level matches harder to surface.

---

## 0. Problem Analysis

### 0.1 Query Noise

The Universal Fallback in `app.py` passes `transcribed_text` directly to
`context_search.search()`. Typical fallback queries look like:

| Raw Transcription | Actual Topic | Noise Words |
|---|---|---|
| `"move to the RISC slide"` | `RISC` | `move to the … slide` |
| `"show me the Harvard architecture"` | `Harvard architecture` | `show me the` |
| `"let's look at the CISC stuff"` | `CISC` | `let's look at the … stuff` |
| `"go back to the pipelining part"` | `pipelining` | `go back to the … part` |
| `"where was the Von Neumann thing"` | `Von Neumann` | `where was the … thing` |

The all-MiniLM-L6-v2 model encodes the **entire** input string into a single
384-dimensional vector. Navigational filler words shift the vector away from the
pure topic embedding, reducing cosine similarity against slide content.

### 0.2 Semantic Dilution in the Index

Current `_extract_slide_text()` flattens **everything** into one string:

```
"Harvard Architecture Physically separate storage for instructions and data
CPU can access instructions and read/write data simultaneously …"
```

For the real slides in `data/slides.json`, slide 2 ("Harvard Architecture") has
a heading of 2 words and 5 bullet items totaling ~60 words. The heading's signal
is drowned out by the body mass. When a user says just *"Harvard architecture"*,
the query embedding matches better against a concise heading embedding than
against the diluted full-text embedding.

**Evidence from Phase 5 debugging**: Embedding scores for topic queries against
flattened slide text maxed at 0.26–0.42 — barely above the 0.30 threshold. A
dual-index approach where titles are embedded separately will produce higher
scores for heading-level matches.

---

## 1. Query Cleaning (`_clean_query`)

### 1.1 Function Signature

```python
def _clean_query(query: str) -> str:
```

Internal helper called at the top of `search()` before embedding the query.

### 1.2 Cleaning Strategy

A single `re.sub` pass using an alternation pattern that strips common
navigational phrases. The regex operates on the lowercased query, then the
result is stripped of extra whitespace.

**Regex pattern** (compiled at module level for performance):

```python
_NAV_NOISE_RE = re.compile(
    r"\b(?:"
    r"(?:can you |could you |please |let's |let us |i want to |i'd like to )?"
    r"(?:move to|go to|go back to|jump to|skip to|switch to|take me to|"
    r"show me|find|where is|where was|where did we (?:discuss|talk about|cover))"
    r")\b"
    r"|\b(?:the )?(?:slide|part|section|bit|stuff|thing|one)"
    r"(?:\s+(?:about|on|with|for|where))?\b",
    re.IGNORECASE,
)
```

This handles two categories:

**Category A — Navigational verb phrases** (optional polite prefix + nav verb):
- `move to`, `go to`, `go back to`, `jump to`, `skip to`, `switch to`
- `take me to`, `show me`, `find`
- `where is`, `where was`, `where did we discuss/talk about/cover`
- Optional prefixes: `can you`, `could you`, `please`, `let's`, `let us`,
  `i want to`, `i'd like to`

**Category B — Filler nouns** (with optional trailing preposition):
- `the slide`, `the part`, `the section`, `the bit`, `the stuff`, `the thing`,
  `the one` — optionally followed by `about`, `on`, `with`, `for`, `where`
- Also matches without `the`: `slide about`, `part about`, etc.

### 1.3 Post-Regex Cleanup

After regex substitution, collapse multiple spaces and strip:

```python
def _clean_query(query: str) -> str:
    cleaned = _NAV_NOISE_RE.sub("", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
```

### 1.4 Examples

| Raw Query | After Cleaning |
|---|---|
| `"move to the RISC slide"` | `"RISC"` |
| `"show me the Harvard architecture"` | `"Harvard architecture"` |
| `"let's look at the CISC stuff"` | `"let's look at CISC"` |
| `"go back to the pipelining part"` | `"pipelining"` |
| `"where was the Von Neumann thing"` | `"Von Neumann"` |
| `"CISC architecture"` | `"CISC architecture"` (no noise to remove) |
| `"the part about registers"` | `"registers"` |
| `"can you show me the slide about revenue"` | `"revenue"` |

### 1.5 Edge Case: Empty String After Cleaning

If the cleaning strips everything (e.g., user says *"go to the slide"* with no
topic), the cleaned query is empty. In this case, `search()` falls back to using
the **original raw query** as-is:

```python
def search(query: str, threshold: float = SEARCH_THRESHOLD) -> dict | None:
    ...
    cleaned = _clean_query(query)
    search_query = cleaned if cleaned else query
    query_embedding = encode([search_query])
    ...
```

This ensures we never pass an empty string to `encode()` (which would produce a
meaningless zero-like vector), while still giving the raw query a chance to match
on its own — even if poorly.

---

## 2. Dual-Embedding Index (Title vs. Body)

### 2.1 New Module State

Replace the single `_slide_embeddings` tensor with two separate tensors:

```python
# ── Module State ─────────────────────────────────────────────────────────────

_title_embeddings = None   # tensor (num_slides × 384), or None
_body_embeddings = None    # tensor (num_slides × 384), or None
_slide_titles: list[str] = []
_slide_bodies: list[str] = []
_num_slides: int = 0
```

Remove the old `_slide_embeddings` and `_slide_texts` variables entirely.

### 2.2 New Text Extraction Helpers

Split `_extract_slide_text()` into two functions:

```python
def _extract_title(slide: dict) -> str:
    """Extract the title/heading from a slide."""
    parts: list[str] = []
    for key in ("heading", "subheading"):
        val = slide.get(key)
        if val:
            parts.append(val)
    return " ".join(parts)


def _extract_body(slide: dict) -> str:
    """Extract body content (bullets, columns, captions, quotes, notes)."""
    parts: list[str] = []
    for key in ("body", "caption", "quote", "attribution"):
        val = slide.get(key)
        if val:
            parts.append(val)
    for item in slide.get("items", []):
        parts.append(item)
    for side in ("left", "right"):
        col = slide.get(side)
        if isinstance(col, dict):
            if col.get("title"):
                parts.append(col["title"])
            for item in col.get("items", []):
                parts.append(item)
    if slide.get("notes"):
        parts.append(slide["notes"])
    return " ".join(parts)
```

The old `_extract_slide_text()` function is **removed** — its responsibilities
are fully covered by the two new helpers.

### 2.3 Refactored `build_index()`

```python
def build_index(slides: list[dict]) -> None:
    global _title_embeddings, _body_embeddings, _slide_titles, _slide_bodies, _num_slides

    _slide_titles = [_extract_title(s) for s in slides]
    _slide_bodies = [_extract_body(s) for s in slides]
    _num_slides = len(slides)

    # Encode titles — use heading text, fall back to " " for title-less slides
    title_texts = [t if t.strip() else " " for t in _slide_titles]
    _title_embeddings = encode(title_texts) if _num_slides > 0 else None

    # Encode bodies — use body text, fall back to " " for body-less slides
    body_texts = [b if b.strip() else " " for b in _slide_bodies]
    _body_embeddings = encode(body_texts) if _num_slides > 0 else None

    if _num_slides > 0:
        logger.info("Context index built: %d slides (title + body).", _num_slides)
    else:
        logger.warning("No slides to index — context search will be empty.")
```

### 2.4 Edge Case: Missing Title or Body

Several slides in the real dataset have generic headings like `"Slide 1"`,
`"Slide 4"`, `"Slide 6"` (these are image-only slides with no meaningful title).
Others have titles but empty `items` lists (no body content).

**Strategy**: Replace empty strings with a single space `" "` before encoding.
This ensures:
- The tensor shapes are consistent (every slide has exactly one title embedding
  and one body embedding).
- `encode()` never receives an empty string (which could produce unstable
  embeddings depending on the tokenizer).
- The `" "` embedding will have very low cosine similarity to any real query,
  so it effectively acts as a "no match" for that channel — it won't produce
  false positives.

**Why not skip empty entries**: Skipping would make the embedding tensor indices
misalign with the slide indices. We'd need a separate index mapping, adding
complexity for no benefit. A space placeholder is simpler and equally effective.

---

## 3. Max-Score Search Logic

### 3.1 Updated `search()` Flow

```python
def search(query: str, threshold: float = SEARCH_THRESHOLD) -> dict | None:
    if _title_embeddings is None or _num_slides == 0:
        return None

    # Step 1: Clean the query
    cleaned = _clean_query(query)
    search_query = cleaned if cleaned else query

    # Step 2: Encode once
    query_embedding = encode([search_query])

    # Step 3: Search both channels
    title_hits = util.semantic_search(query_embedding, _title_embeddings, top_k=1)
    body_hits = util.semantic_search(query_embedding, _body_embeddings, top_k=1)

    # Step 4: Pick the absolute best score across both channels
    best_title = title_hits[0][0] if title_hits and title_hits[0] else None
    best_body = body_hits[0][0] if body_hits and body_hits[0] else None

    candidates = []
    if best_title:
        candidates.append(best_title)
    if best_body:
        candidates.append(best_body)

    if not candidates:
        return None

    best = max(candidates, key=lambda h: h["score"])

    if best["score"] >= threshold:
        idx = best["corpus_id"]
        matched_text = _slide_titles[idx] or _slide_bodies[idx]
        logger.info(
            "Context search: '%.60s' → slide %d (score: %.2f)",
            search_query, idx, best["score"],
        )
        return {
            "slide_index": idx,
            "score": float(best["score"]),
            "matched_text": matched_text[:100],
        }

    logger.debug("Context search: no match above %.2f for '%.60s'", threshold, search_query)
    return None
```

### 3.2 Key Design Decisions

| Decision | Rationale |
|---|---|
| Encode the query **once**, search **both** tensors | One `encode()` call (~2–5 ms) vs. two. The `semantic_search` calls are pure tensor ops (~0.1 ms each) — negligible. |
| Take `max(title_score, body_score)` | A short topic query like `"RISC"` will score much higher against the concise title `"Features of RISC Processors"` than the 60-word body. A detailed query like `"pipelining for better efficiency"` might score higher against the body. Taking the max lets both channels contribute without needing separate thresholds. |
| `matched_text` uses title preferentially | For the response payload, the title is more useful as a human-readable summary. Falls back to body text if the title is empty. |
| `SEARCH_THRESHOLD` stays at `0.30` | With query cleaning and title-level matching, scores will increase. We keep the threshold at 0.30 to remain inclusive — the user can always tune it up later. No need to preemptively tighten. |
| The `" "` placeholder for empty title/body means `corpus_id` alignment is guaranteed | Both tensors have exactly `_num_slides` rows. `corpus_id` from `semantic_search` directly maps to the slide index — no translation needed. |

### 3.3 What Stays Unchanged

- `SEARCH_THRESHOLD = 0.30`
- The `encode()` call from `embeddings.py` (shared singleton)
- The return format: `{"slide_index": int, "score": float, "matched_text": str}`
- How `app.py` calls `context_search.search()` — the interface is identical
- How `app.py` calls `context_search.build_index()` — same `list[dict]` input

---

## 4. Testing Strategy (`test_context_search.py`)

### 4.1 Tests to Delete

| Test | Reason |
|---|---|
| `test_extract_text_all_slide_types` | Tests `_extract_slide_text()` which no longer exists. Replaced by separate title/body extraction tests. |

### 4.2 Tests to Update

| Test | Change |
|---|---|
| `test_build_index_populates_embeddings` | Assert `_title_embeddings` and `_body_embeddings` are not None (instead of `_slide_embeddings`). Assert `_num_slides`. |
| `test_build_index_empty_slides` | Assert `_title_embeddings` is None (instead of `_slide_embeddings`). |
| `test_reindex_updates_embeddings` | No assertion changes — it tests `search()` results, which still work the same way. |

### 4.3 Tests to Keep Unchanged

These test `search()` behavior through the public API and remain valid:

| Test | Why It Stays |
|---|---|
| `test_search_exact_heading_match` | `"Harvard Architecture"` should still match slide 1. With title-level embeddings, the score should actually *increase*. |
| `test_search_partial_topic_match` | `"RISC processors pipelining"` should still match slide 2. Body embeddings cover this. |
| `test_search_no_match_below_threshold` | `"quantum physics dark matter"` should still return None. |
| `test_search_empty_index` | Empty index should still return None. |
| `test_reindex_updates_embeddings` | Re-indexing should still work. |

### 4.4 New Tests — Query Cleaning

| Test Name | Input | Expected Output | What It Proves |
|---|---|---|---|
| `test_clean_query_strips_nav_verbs` | `"move to the RISC slide"` | `"RISC"` | Nav verb + filler noun removed, topic preserved |
| `test_clean_query_strips_show_me` | `"show me the Harvard architecture"` | `"Harvard architecture"` | `show me the` removed, multi-word topic preserved |
| `test_clean_query_strips_compound_phrase` | `"can you take me to the part about pipelining"` | `"pipelining"` | Polite prefix + nav verb + filler noun + preposition removed |
| `test_clean_query_preserves_bare_topic` | `"CISC architecture"` | `"CISC architecture"` | No noise present — query passes through unchanged |
| `test_clean_query_empty_result_fallback` | `"go to the slide"` | `""` (empty string) | All words are noise — returns empty, `search()` will fall back to raw query |
| `test_clean_query_strips_where_was` | `"where was the Von Neumann thing"` | `"Von Neumann"` | `where was the` + `thing` removed |

### 4.5 New Tests — Dual-Index Text Extraction

| Test Name | Input Slide | Expected |
|---|---|---|
| `test_extract_title_heading_and_subheading` | `{"heading": "Q3 Review", "subheading": "By Jane"}` | `"Q3 Review By Jane"` |
| `test_extract_title_heading_only` | `{"heading": "Harvard Architecture"}` | `"Harvard Architecture"` |
| `test_extract_title_empty` | `{"heading": "", "items": ["bullet"]}` | `""` (empty string) |
| `test_extract_body_bullets` | `{"heading": "Title", "items": ["a", "b"]}` | `"a b"` (heading excluded from body) |
| `test_extract_body_two_column` | Two-column slide from `SAMPLE_SLIDES` | Contains `"Engineering"`, `"Product"`, column items |
| `test_extract_body_empty` | `{"heading": "Title Only"}` | `""` (no body content) |

### 4.6 New Tests — Dual-Index Build & Search

| Test Name | Setup | Assertion | What It Proves |
|---|---|---|---|
| `test_build_index_dual_tensors` | `build_index(SAMPLE_SLIDES)` | `_title_embeddings.shape[0] == 6` and `_body_embeddings.shape[0] == 6` | Both tensors have correct number of rows |
| `test_title_match_beats_body_dilution` | Build index with a slide that has a specific heading `"Pipelining"` and a long body. Search for `"pipelining"`. | Score against title-only index > score against body-only index for the same slide. | Title embeddings produce higher scores for concise topic queries |
| `test_body_match_works_for_detail_query` | Build index with SAMPLE_SLIDES. Search for `"separate storage for instructions and data"`. | Matches slide 1 (Harvard Architecture — this phrase is in the body bullets). | Body embeddings still catch detailed, content-specific queries |
| `test_missing_title_no_crash` | Build index with `[{"heading": "", "items": ["bullet text"]}]`. Search for `"bullet text"`. | Returns a match (via body channel). No crash. | Empty title placeholder works correctly |
| `test_missing_body_no_crash` | Build index with `[{"heading": "Solo Title"}]`. Search for `"Solo Title"`. | Returns a match (via title channel). No crash. | Empty body placeholder works correctly |
| `test_noisy_query_finds_correct_slide` | Build index with SAMPLE_SLIDES. Search for `"move to the RISC slide"`. | Returns slide 2 (RISC). | End-to-end: query cleaning + title matching produces correct result |

### 4.7 Test Count Summary

| Category | Before | After |
|---|---|---|
| Text extraction | 1 | 6 (title + body) |
| build_index | 2 | 3 (+ dual tensor test) |
| search (existing) | 5 | 5 (kept) |
| Query cleaning | 0 | 6 (new) |
| Dual-index edge cases | 0 | 5 (new) |
| **Total** | **8** | **25** |

---

## 5. Files Changed — Summary

| File | Action | Net Change |
|---|---|---|
| `backend/context_search.py` | **Modify** | Add `import re`, `_clean_query()`, `_NAV_NOISE_RE`. Replace `_extract_slide_text()` with `_extract_title()` + `_extract_body()`. Replace `_slide_embeddings`/`_slide_texts` with `_title_embeddings`/`_body_embeddings`/`_slide_titles`/`_slide_bodies`. Refactor `build_index()` for dual encoding. Refactor `search()` for cleaning + dual-channel max-score. ~45 lines removed, ~70 lines added. |
| `tests/test_context_search.py` | **Modify** | Delete 1 old extraction test (~30 lines). Update 2 build_index tests. Add 17 new tests (~120 lines). |

**No changes to**: `app.py`, `intent_classifier.py`, `embeddings.py`, `config.py`,
`app.js`, or any other file. The `search()` and `build_index()` public interfaces
remain identical — all callers work unchanged.

---

## 6. Execution Order

1. **`backend/context_search.py`** — Add `_clean_query`, replace extraction helpers,
   refactor `build_index()` and `search()`.
2. **`tests/test_context_search.py`** — Delete old extraction test, update build_index
   tests, add 17 new tests.
3. **Run all tests** — Verify the full suite passes (intent classifier, context search,
   universal fallback, transcriber).

---

## 7. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Regex over-strips topic words that happen to match nav patterns | User says `"go to the section on products"` — `"section on"` stripped, but `"products"` preserved | The regex only strips the navigational framing, not content words. `"products"` survives. Edge cases can be addressed by tightening word boundaries. |
| `" "` placeholder produces a false-positive match | A space-only embedding scores high enough to match some query | Extremely unlikely — a space tokenizes to the `[CLS] [SEP]` tokens only, producing a near-zero-magnitude vector. Cosine similarity against any real query will be near 0.0, well below the 0.30 threshold. |
| Two `semantic_search` calls instead of one doubles search time | Increased latency | Each `semantic_search` call on a 13-slide corpus takes ~0.1 ms. Two calls = ~0.2 ms. The bottleneck is `encode()` at ~2–5 ms, which only runs once. Total latency increase is negligible. |
| Two tensors double VRAM for the index | Increased memory usage | Each tensor for 13 slides × 384 dims × 4 bytes = ~20 KB. Doubling to ~40 KB is negligible. Even for 1,000 slides, it would be ~3 MB — trivial compared to the 90 MB model. |
| `_extract_slide_text()` removal breaks imports in test file | Test import error | The test file currently imports `_extract_slide_text`. This import will be updated to import `_extract_title` and `_extract_body` instead. |
