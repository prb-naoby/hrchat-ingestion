# LLM Document Ingestion API

This service exposes a FastAPI application that ingests documents into Qdrant and surfaces retrieval helpers.

## Environment

Set the following variables (or copy `.env.example`):

- `GOOGLE_API_KEY` / `GENAI_API_KEY` – Gemini access.
- `QDRANT_URL` (+ optional `QDRANT_API_KEY`) – Qdrant endpoint.
- `QDRANT_COLLECTION` – default collection name.

Install requirements and run:

```bash
pip install -r requirements.txt
uvicorn llm_document_ingestion.src.api:app --reload
```

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/process` | POST (multipart) | Upload a PDF/image with `category` + `collection`. The pipeline parses, chunks, embeds, and writes to Qdrant. Duplicate filenames are skipped via job markers. |
| `/changes` | POST (JSON) | Accepts a OneDrive inventory payload. Returns per-category lists of `removed` and `new_or_updated` files based on Qdrant state. |
| `/retrieve` | POST | Vector retrieval over stored embeddings (see `src/retrieve.py` for parameters). |
| `/filenames` | GET | Return unique filenames across all points for a given `category`, using the default Qdrant collection. |
| `/bm25/*` | GET/POST | Sparse/BM25 helpers enabled when the collection has BM25 configured. |

`/process` accepts an `async_mode` query param (`1` = background task, default; `0` = synchronous).

## Components

- `src/api.py` – FastAPI app wiring all routes above.
- `src/pipeline_sync.py` — parsing, chunking, embedding, Qdrant upsert.
- `src/qdrant_utils.py` — Qdrant client helper and marker utilities.
- `src/logger.py` — logging adapter shared by the service.

## Parse progress & retries

- The parser persists incremental checkpoints (base markdown, figure metadata, and Gemini outputs) in the queue cache directory. Heartbeats are written every ~25 pages / few figure descriptions so long-running Docling runs expose progress via logs.
- Queue metadata (`*.pdf.json`) now mirrors the latest heartbeat under `parse_progress`, making it easy to see the active phase and last update timestamp from disk or `/queue/failed`.
- If a worker crashes mid-parse, the next attempt resumes from the cached checkpoint rather than restarting Docling/VLM calls. Stale heartbeats (>15 minutes) are logged when a job is reclaimed so operators know a resume is happening.
- When a PDF has a known page count, Docling slices the work into batches sized at `PARSE_PAGE_BATCH_PERCENT` of the document (default 5%) but never exceeding `PARSE_PAGE_BATCH_SIZE` (default 50 pages). When the page count is unknown it falls back to that fixed size. Tuning these variables lets you balance responsiveness versus throughput while keeping resume capability between batches.
- `/queue/status` exposes live queue jobs (queued and processing) along with their parse heartbeat metadata, batch counters, and last-updated timestamps so dashboards no longer need to peek at the sidecar files.
