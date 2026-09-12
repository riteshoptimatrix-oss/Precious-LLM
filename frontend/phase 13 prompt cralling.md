# Phase 13: Website Knowledge Engine, Crawler & Chatbot Integration

## Overview

After a full audit of the existing codebase, the **website sub-package skeleton is already in place** across all five sub-directories (`crawler/`, `extraction/`, `processing/`, `search/`, `repositories/`). Phase 13 execution has two distinct jobs:

1. **Fix existing bugs** found in the already-written skeleton files.
2. **Build the missing layer** — the services layer, all core-module extensions, CLI scripts, tests, and health endpoint — that wires the website engine into the live chatbot pipeline.

---

## Existing Code Audit: Bugs Found

> [!CAUTION]
> Three import/syntax bugs in the already-committed skeleton files **will crash the application at import time**. These must be fixed before building anything new.

| File | Line | Bug | Fix |
|---|---|---|---|
| [`rate_limiter.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/website/crawler/rate_limiter.py) | 22 | `async def acquire((self))` — double-paren syntax error | `async def acquire(self)` |
| [`fetcher.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/website/crawler/fetcher.py) | 1-16 | `asyncio.sleep()` called but `asyncio` never imported | Add `import asyncio` |
| [`boilerplate.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/website/extraction/boilerplate.py) | 20 | `Optional` used in `__init__` signature but never imported | Add `from typing import Optional` |

---

## What Already Exists (No Action Needed)

| Module | Status |
|---|---|
| `crawler/` — all 7 modules (crawler, fetcher, fetcher, models, rate_limiter, robots, url_discovery, url_normalizer) | ✅ Complete (bugs fixed above) |
| `extraction/` — all 4 modules (html_parser, cleaner, boilerplate, classifier) | ✅ Complete (bug fixed above) |
| `processing/` — chunker, hasher, validator | ✅ Complete |
| `search/` — query, scorer, search_engine | ✅ Complete |
| `repositories/` — all 4 repos (page, chunk, version, crawl_run) | ✅ Complete |

---

## What Must Be Built (Phase 13 Execution Scope)

### Layer 1 — Processing: `versioning.py` (missing module)

#### [NEW] `versioning.py`
Path: `backend/app/website/processing/versioning.py`

Implements `WebsiteVersionManager` — orchestrates atomic dataset activation and rollback by coordinating `WebsitePageRepository`, `WebsiteChunkRepository`, and `WebsiteVersionRepository` in a single transaction-safe sequence.

---

### Layer 2 — Services (entire directory missing)

The `backend/app/website/services/` directory does not exist. All three service files must be created.

#### [NEW] `__init__.py`
Path: `backend/app/website/services/__init__.py`

#### [NEW] `crawl_service.py`
Path: `backend/app/website/services/crawl_service.py`

Crawl orchestrator that:
- Instantiates `WebCrawler`, `HTMLDocumentParser`, `WebsiteChunker`, `ContentHasher`
- Runs full crawl loop: fetch → parse → hash → chunk → persist pages → persist chunks → activate version → save crawl run log
- Supports `dry_run=True` mode that discovers URLs without writing to MongoDB
- Returns `CrawlRunSummary` with full statistics

#### [NEW] `website_knowledge_service.py`
Path: `backend/app/website/services/website_knowledge_service.py`

Search facade + health reporter that:
- Wraps `WebsiteSearchEngine` and formats results into a context string block (`<website_knowledge>…</website_knowledge>`)
- Provides `get_health_status()` returning version, chunk count, page count, last crawl run timestamp

#### [NEW] `website_refresh_service.py`
Path: `backend/app/website/services/website_refresh_service.py`

Incremental refresh service that:
- Loads existing page hashes from MongoDB before crawling
- Classifies each fetched page as `new`, `changed`, or `unchanged` using SHA-256 comparison
- Only re-chunks and re-indexes changed/new pages
- Supports explicit version rollback via `rollback_to_version(version_id)`

---

### Layer 3 — Core Module Extensions

#### [MODIFY] [`router.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/knowledge/router.py)

Extend `RoutingDecision` enum with two new values:
```python
WEBSITE_KNOWLEDGE_REQUIRED = "WEBSITE_KNOWLEDGE_REQUIRED"
WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED = "WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED"
```

Add `WEBSITE_TRIGGERS` keyword list covering: services, countries, admissions, fees, visa guidance, study abroad, IELTS/TOEFL, counselling, preciousedu.in website details, contact/office/branch.

Extend `route()` with a 4th decision step:
- If website triggers match but no project triggers → `WEBSITE_KNOWLEDGE_REQUIRED`
- If both sets trigger → `WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED`
- Existing project-only logic unchanged

#### [MODIFY] [`engine.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/knowledge/engine.py)

Add two new methods:
```python
async def retrieve_website_context(user_message, recent_messages) -> Tuple[RoutingDecision, Optional[str]]
async def retrieve_combined_context(user_message, recent_messages) -> Tuple[RoutingDecision, Optional[str], Optional[str]]
```

`KnowledgeEngine.__init__` accepts optional `website_knowledge_service` instance (injected from `ConversationEngine`).

#### [MODIFY] [`context_builder.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/conversation/context_builder.py)

Add field to `ConversationContext`:
```python
website_knowledge: Optional[str] = Field(None, description="Retrieved website knowledge context block")
```

Add `website_knowledge` parameter to `build_context()` signature.

#### [MODIFY] [`engine.py` (ConversationEngine)](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/conversation/engine.py)

- Accept optional `website_knowledge_service` in `__init__`; default-construct `WebsiteKnowledgeService(db)` if not injected
- After step 6 (project knowledge retrieval), add step 6b: call router; if `WEBSITE_KNOWLEDGE_REQUIRED` or `WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED`, call `website_knowledge_service.search()`
- Pass `website_knowledge=web_context_text` into `context_builder.build_context()`
- Add `sources` list to response metadata (URL + page title pairs from website chunks used)

#### [MODIFY] [`service.py` (LLMService)](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/llm/service.py)

In `_serialize_context()`, after reading `project_knowledge`, also read `website_knowledge`:
```python
website_knowledge = getattr(context, "website_knowledge", None)
```

Include `<website_knowledge>\n{website_knowledge}` block in prompt assembly, placed between `<project_knowledge>` (if any) and `<user>` tags. Context budget calculation updated to account for this additional block.

#### [MODIFY] [`chat.py` (schemas)](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/schemas/chat.py)

Add optional field to `ChatResponse`:
```python
sources: List[Dict[str, str]] = Field(default_factory=list, description="Website source references used")
```

#### [MODIFY] [`health.py`](file:///c:/Users/Lenovo/Desktop/Ritesh%20Gajjar%20Work/Precious%20Edu%20LLM/backend/app/api/routes/health.py)

Add new endpoint:
```python
GET /api/health/website
```

Returns: active version ID, chunk count, page count, last crawl datetime, `is_ready` boolean flag.

---

### Layer 4 — CLI Management Scripts (all new)

All scripts live in `backend/scripts/` and use `asyncio.run()` + `argparse`.

| Script | Purpose |
|---|---|
| `crawl_website.py` | Full or `--dry-run` website crawl |
| `refresh_website.py` | Incremental refresh (changed pages only) |
| `inspect_website.py` | Print active version stats, chunk count, last crawl run |
| `search_website.py` | Test retrieval: `python -m scripts.search_website "study visa"` |
| `activate_website_version.py` | Activate a specific version ID or rollback |
| `evaluate_website_knowledge.py` | Run Q&A test suite; emit precision/recall/source-grounding metrics |

---

### Layer 5 — Tests (`backend/tests/website/`)

The `website/` test directory does not yet exist. Create:

```
backend/tests/website/
├── __init__.py
├── test_url_normalizer.py       # Domain restriction, SSRF, fragment strip, tracking params
├── test_robots.py               # robots.txt fetch, allow/disallow compliance
├── test_url_discovery.py        # HTML link extraction, sitemap parsing
├── test_rate_limiter.py         # Delay enforcement, concurrency slots
├── test_html_parser.py          # Title, headings, section extraction, Unicode
├── test_cleaner.py              # Script/style/nav removal, Unicode preservation
├── test_boilerplate.py          # Boilerplate line filtering
├── test_classifier.py           # Page type classification rules
├── test_chunker.py              # Chunk word bounds, overlap, hash generation
├── test_hasher.py               # SHA-256 determinism and empty-string handling
├── test_validator.py            # Empty-page ratio, duplicate hash detection
├── test_versioning.py           # Atomic activation, rollback mechanics
├── test_query_normalizer.py     # Token normalization, synonym expansion
├── test_scorer.py               # Title/heading/content weighting, exact phrase boost
├── test_search_engine.py        # Top-K retrieval, deduplication, min-score filtering
├── test_knowledge_router.py     # Website trigger routing, combined routing decision
└── test_website_e2e.py          # End-to-end: crawl → index → search → chat integration
```

---

## `requirements.txt` Update

Add `beautifulsoup4>=4.12.0` which is used by extraction modules but is currently absent from `requirements.txt`.

---

## Execution Order

```
1. Fix 3 skeleton bugs (rate_limiter, fetcher, boilerplate)
2. Add beautifulsoup4 to requirements.txt
3. Create processing/versioning.py
4. Create services/__init__.py, crawl_service.py, website_knowledge_service.py, website_refresh_service.py
5. Extend knowledge/router.py (RoutingDecision + WEBSITE_TRIGGERS + route() logic)
6. Extend knowledge/engine.py (retrieve_website_context, retrieve_combined_context)
7. Extend conversation/context_builder.py (website_knowledge field)
8. Extend conversation/engine.py (website knowledge retrieval + sources metadata)
9. Extend llm/service.py (_serialize_context website_knowledge block)
10. Extend schemas/chat.py (sources field)
11. Add /api/health/website endpoint to health.py
12. Create 6 CLI scripts
13. Create tests/website/ with 17 test files
14. Run pytest suite
```

---

## Verification Plan

### Automated Tests
```bash
# Website unit tests
pytest backend/tests/website/ -v

# Full regression suite (Phase 1–13)
pytest backend/tests/ -v --tb=short
```

### Manual Verification
```bash
# 1. Dry-run crawl
python -m scripts.crawl_website --dry-run

# 2. Full crawl
python -m scripts.crawl_website

# 3. Search test
python -m scripts.search_website "study visa Canada fees"

# 4. Health endpoint
curl http://localhost:8000/api/health/website

# 5. Evaluation report
python -m scripts.evaluate_website_knowledge
```

---

## Open Questions

> [!IMPORTANT]
> **1. `BeautifulSoup4` dependency**: The extraction modules import `bs4` but it is absent from `requirements.txt`. Confirm it should be added (almost certainly yes).

> [!IMPORTANT]
> **2. `WebsiteKnowledgeService` injection into `ConversationEngine`**: Should the service be constructed lazily (default-constructed inside `ConversationEngine.__init__`) or require explicit DI from `main.py` application startup? Lazy construction is simpler; explicit DI is more testable and allows startup-time DB index creation.

> [!IMPORTANT]
> **3. Website routing priority**: When both project triggers AND website triggers match, should the router return `WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED` (both retrieved) or `PROJECT_KNOWLEDGE_REQUIRED` (project takes priority)? The plan defaults to retrieving both and including both context blocks.

> [!NOTE]
> **4. `sources` in `ChatResponse`**: The `sources` field (page URL + title pairs) will only be populated for responses that used website knowledge. Non-website turns will return `sources: []`. This is backward-compatible.
