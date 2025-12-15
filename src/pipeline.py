"""Synchronous document ingestion pipeline.

This module orchestrates the end-to-end ingestion flow: parsing PDFs or
images, cleaning and chunking the resulting markdown, embedding chunks
and writing them into Qdrant. All operations are synchronous; there
are no ``async def`` functions or semaphores. Concurrency is used only
within the parser to describe figures via Gemini (see
``parser_utils._describe_figures``).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterable, List, Optional
from .parser_utils import parse_document
from .chunk import chunk_markdown
from .embed import get_genai_embedding
from .qdrant_utils import qdrant_client, ensure_collection, marker_point, build_payload
from .logger import get_logger, EMBED_BATCH_SIZE, CHUNK_MAX_CHARS
from .parse_cache import ParseProgressTracker


def _batched(seq: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive ``size``-length lists from ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def process_file(
    *,
    data: Optional[bytes] = None,
    file_path: Optional[str] = None,
    filename: str,
    ext: str,
    category: str,
    collection: str,
    job_key: str,
    cache_path: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run the synchronous ingestion pipeline over a document."""
    if data is None and file_path is None:
        raise ValueError("process_file requires either 'data' or 'file_path'")
    log = get_logger(job=job_key, file=filename, phase="process")
    t_overall = time.perf_counter()

    # Base identifier (filename without extension) for payload display & grouping
    doc_id = os.path.splitext(filename)[0]

    # -------- PARSE --------
    markdown: str
    metadata: Dict[str, Any]
    cache_loaded = False
    parse_log = log.with_phase("parse")
    tracker: Optional[ParseProgressTracker] = None

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tracker = ParseProgressTracker(
            cache_file,
            parse_log,
            progress_callback=progress_callback,
        )
        if tracker.has_final():
            markdown, metadata = tracker.get_final()
            cache_loaded = True
            parse_log.info(
                "Loaded cached parse result "
                f"(size_bytes={metadata.get('file_size_bytes')} pages={metadata.get('page_count')})"
            )
        elif tracker.doc_ready():
            parse_log.info("Found cached Docling checkpoint; resuming parse from cache")

    if not cache_loaded:
        if ext == ".pdf":
            tmp_path = file_path
            remove_tmp = False
            if tmp_path is None:
                if data is None:
                    raise ValueError("PDF ingestion requires either 'data' bytes or 'file_path'")
                with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                remove_tmp = True
            try:
                markdown, metadata = parse_document(
                    tmp_path,
                    filename,
                    ext,
                    job_key=job_key,
                    cache_tracker=tracker,
                )
            finally:
                if remove_tmp and tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        else:
            if data is None:
                try:
                    if file_path is None:
                        raise ValueError("Non-PDF ingestion requires 'data' or 'file_path'")
                    with open(file_path, "rb") as fh:
                        data = fh.read()
                except Exception as exc:
                    raise RuntimeError(f"Failed to load image bytes from {file_path}: {exc}") from exc
            markdown, metadata = parse_document(
                "",
                filename,
                ext,
                image_bytes=data,
                job_key=job_key,
                cache_tracker=tracker,
            )
        parse_log.info(
            f"Completed parse stage in {time.perf_counter() - t_overall:.3f}s"
        )
        if tracker:
            tracker.finalize(markdown, metadata)

    # -------- CHUNK --------
    t_chunk = time.perf_counter()
    log.with_phase("chunk").info("Starting chunk stage")
    chunks: List[str] = chunk_markdown(markdown, max_chars=CHUNK_MAX_CHARS)
    expected = len(chunks)
    log.with_phase("chunk").info(
        f"Completed chunk stage count={expected} in {time.perf_counter() - t_chunk:.3f}s"
    )

    uploaded = 0

    # -------- EMBED & UPSERT --------
    has_qdrant = False
    try:
        client = qdrant_client()
        has_qdrant = True
    except Exception as exc:
        client = None  # type: ignore
        log.with_phase("embed").warning(f"Skipping Qdrant upsert due to client init failure: {exc}")

    marker_vec: Optional[List[float]] = None
    if has_qdrant and chunks:
        # Prepare marker vector and ensure collection exists.
        marker_vec = get_genai_embedding("job_marker")
        ensure_collection(client, collection, len(marker_vec))

        # Create payload indexes for downstream filtering.
        for field_name in ("metadata.filename", "metadata.category", "type"):
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema="keyword",
                    wait=True,
                )
            except Exception:
                pass

        # Initial job marker (ID is UUIDv5 via marker_point; filename kept in payload)
        client.upsert(
            collection_name=collection,
            points=[
                marker_point(
                    job_key=job_key,
                    vector=marker_vec,
                    filename=filename,   # pass original; helper strips ext
                    category=category,
                    collection=collection,
                    status="running",
                    expected_chunks=expected,
                    uploaded_chunks=uploaded,
                )
            ],
            wait=True,
        )
        log.with_phase("embed").info(
            f"Started embedding: total_chunks={expected} batch_size={EMBED_BATCH_SIZE}"
        )

        # Embed & upsert in small batches
        chunk_index_offset = 0
        for batch in _batched(chunks, EMBED_BATCH_SIZE):
            t_batch = time.perf_counter()

            # 1) Embed (sequential)
            vectors = [get_genai_embedding(text) for text in batch]

            # 2) Build points (chunk IDs are random UUIDs; filename shown in payload)
            points: List[Dict[str, Any]] = []
            for i, (text_chunk, vec) in enumerate(zip(batch, vectors)):
                idx = chunk_index_offset + i
                payload = build_payload(
                    text=text_chunk,
                    filename=doc_id,
                    category=category,
                    chunk_index=idx,
                    dim=len(vec),
                    type="chunk",
                )
                points.append(
                    {
                        "id": uuid.uuid4().hex,  # valid UUID (simple form)
                        "vector": vec,
                        "payload": payload,
                    }
                )

            # 3) Upsert batch
            client.upsert(collection_name=collection, points=points, wait=True)
            uploaded += len(points)

            # 4) Update progress marker
            client.upsert(
                collection_name=collection,
                points=[
                    marker_point(
                        job_key=job_key,
                        vector=marker_vec,
                        filename=filename,
                        category=category,
                        collection=collection,
                        status="running",
                        expected_chunks=expected,
                        uploaded_chunks=uploaded,
                    )
                ],
                wait=True,
            )
            log.with_phase("embed").info(
                f"Batch uploaded={len(points)} total_uploaded={uploaded} took={time.perf_counter() - t_batch:.3f}s"
            )
            chunk_index_offset += len(batch)

        # Final marker
        client.upsert(
            collection_name=collection,
            points=[
                marker_point(
                    job_key=job_key,
                    vector=marker_vec,
                    filename=filename,
                    category=category,
                    collection=collection,
                    status="done",
                    expected_chunks=expected,
                    uploaded_chunks=uploaded,
                )
            ],
            wait=True,
        )
        log.with_phase("embed").info(
            f"Completed embedding uploaded={uploaded} overall={time.perf_counter() - t_overall:.3f}s"
        )
    else:
        log.with_phase("embed").warning(
            "No embeddings were uploaded (no Qdrant client or no chunks)"
        )

    total_seconds = time.perf_counter() - t_overall

    # Compose result
    result: Dict[str, Any] = {
        "file_name": filename,
        "file_type": metadata.get("file_type"),
        "page_count": metadata.get("page_count"),
        "images_count": metadata.get("images_count"),
        "uploaded_chunks": uploaded,
        "collection": collection,
        "total_processing_seconds": total_seconds,
        "category": category,
        "markdown": markdown,
    }
    return result


__all__ = [
    "process_file",
]








