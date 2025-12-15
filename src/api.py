"""FastAPI application exposing a single `/process` endpoint.

Flow:
1) Pre-check in Qdrant to avoid duplicate work.
   - Look for existing chunks (metadata.filename + metadata.category) and
     inspect any job markers associated with the document.

2) If already exists:
   - conflict|ok|skip behavior

3) If not exists:
   - Default async enqueue (file saved for later processing); optional sync ingestion if async_mode=0.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

from .pipeline import process_file
from .qdrant_utils import qdrant_client, delete_points_by_filename
from .logger import get_logger
from .sync_utils import (
    CategoryDiff,
    compute_category_diffs,
    drive_items_from_payload,
    group_items_by_category,
)
from .pipeline_queue import run_once as run_queue_once, get_failed_jobs, get_queue_status


app = FastAPI(title="Document Ingestion API")
QUEUE_DIR = Path(os.getenv("INGEST_QUEUE_DIR", "ingest_queue"))

# ---------------------------
# Helpers
# ---------------------------


class ChangesRequest(BaseModel):
    """Request body for the `/changes` endpoint."""

    inventory: Any
    category: Optional[str] = None
    collection: Optional[str] = None


class DeletePointsRequest(BaseModel):
    """Request body for deleting Qdrant points by filename."""

    filename: str
    collection: Optional[str] = None
    category: Optional[str] = None

def _marker_id_for_filename(collection: str, filename_no_ext: str) -> str:
    """Deterministic UUIDv5 marker id based on collection + filename (sans extension)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{filename_no_ext.lower()}"))


def _idempotent_create_indexes(client, collection: str, fields: Iterable[str]) -> None:
    """
    Create keyword payload indexes for given fields (safe to call repeatedly).

    Matches your payload_schema:
      - content
      - metadata.category
      - metadata.filename
      - metadata.chunk_index
      - metadata.dim

    Can optionally be used for additional fields (for example top-level ``type``)
    when callers need to filter by them.
    """
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema="keyword",
                wait=True,
            )
        except Exception as e:
            # ignore "already exists" or transient errors, but log them for debugging
            # get_logger(job="system", file="api", phase="init").debug(f"Index creation note: {e}")
            pass


def _run_ingestion_bg(
    *,
    file_path: str,
    filename: str,
    ext: str,
    category: str,
    collection: str,
    job_key: str,
) -> None:
    """Background ingestion runner; logs failures instead of raising."""
    log = get_logger(job=job_key, file=filename, phase="bg")
    try:
        process_file(
            file_path=file_path,
            filename=filename,
            ext=ext,
            category=category,
            collection=collection,
            job_key=job_key,
        )
        log.info("background ingestion completed")
    except Exception as e:
        log.error(f"background ingestion failed: {e}")
    finally:
        try:
            os.unlink(file_path)
        except Exception:
            pass


# ---------------------------
# FastAPI routes
# ---------------------------


def _format_category_diff(diff: CategoryDiff) -> Dict[str, Any]:
    removed: List[str] = []
    for key in diff.stale_keys:
        record = diff.qdrant_records.get(key)
        base_name = key
        if record and isinstance(record.payload, dict):
            metadata = record.payload.get("metadata") or {}
            if isinstance(metadata, dict):
                base_name = metadata.get("filename", key)
        removed.append(base_name)

    new_or_updated: List[Dict[str, str]] = []
    for key in diff.missing_keys:
        item = diff.item_lookup[key]
        new_or_updated.append(
            {
                "filename": item.raw_name,
                "base_name": item.base_name,
                "download_url": item.download_url,
            }
        )

    return {
        "category": diff.category,
        "onedrive_count": len(diff.drive_keys),
        "qdrant_count": len(diff.qdrant_keys),
        "removed": removed,
        "new_or_updated": new_or_updated,
    }


@app.get("/queue/failed")
def list_failed_jobs_endpoint() -> JSONResponse:
    jobs = get_failed_jobs()
    return JSONResponse(
        {
            "count": len(jobs),
            "jobs": jobs,
        },
        status_code=200,
    )


@app.get("/queue/status")
def list_queue_status_endpoint() -> JSONResponse:
    jobs = get_queue_status()
    return JSONResponse(
        {
            "count": len(jobs),
            "jobs": jobs,
        },
        status_code=200,
    )


@app.post("/qdrant/delete")
def delete_points(payload: DeletePointsRequest) -> JSONResponse:
    """Delete all Qdrant points associated with a filename."""
    try:
        removed = delete_points_by_filename(
            payload.filename,
            collection=payload.collection,
            category=payload.category,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        {
            "filename": payload.filename,
            "collection": payload.collection or os.getenv("QDRANT_COLLECTION", "documents"),
            "category": payload.category,
            "deleted_points": removed,
        },
        status_code=200,
    )


@app.post("/queue/run")
def trigger_queue_run(background_tasks: BackgroundTasks) -> JSONResponse:
    background_tasks.add_task(run_queue_once)
    return JSONResponse({"status": "scheduled"}, status_code=status.HTTP_202_ACCEPTED)


@app.post("/changes")
def list_changes(payload: ChangesRequest) -> JSONResponse:
    """Return per-category document changes between OneDrive and Qdrant."""
    collection = payload.collection or os.getenv("QDRANT_COLLECTION", "documents")

    try:
        items = drive_items_from_payload(
            payload.inventory,
            default_category=payload.category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not items:
        return JSONResponse(
            {
                "collection": collection,
                "categories": [],
                "summary": {"categories": 0, "removed": 0, "new_or_updated": 0},
            },
            status_code=200,
        )

    try:
        client = qdrant_client()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    grouped = group_items_by_category(items)
    diffs = compute_category_diffs(
        client,
        collection=collection,
        grouped_items=grouped,
    )
    categories_data = [_format_category_diff(diff) for diff in diffs]
    summary = {
        "categories": len(categories_data),
        "removed": sum(len(cat["removed"]) for cat in categories_data),
        "new_or_updated": sum(len(cat["new_or_updated"]) for cat in categories_data),
    }
    return JSONResponse(
        {
            "collection": collection,
            "categories": categories_data,
            "summary": summary,
        },
        status_code=200,
    )


@app.get("/filenames")
def list_filenames_by_category(
    category: str = Query(..., min_length=1),
) -> JSONResponse:
    """Return unique filenames in Qdrant for the given category."""
    collection = os.getenv("QDRANT_COLLECTION", "documents")
    try:
        client = qdrant_client()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    _idempotent_create_indexes(
        client,
        collection,
        fields=(
            "metadata.category",
            "metadata.filename",
        ),
    )

    scroll_filter = {
        "must": [
            {"key": "metadata.category", "match": {"value": category}},
        ]
    }

    filenames: set[str] = set()
    offset: Optional[Dict[str, Any]] = None

    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        if not records:
            break
        for record in records:
            metadata = (record.payload or {}).get("metadata") or {}
            if isinstance(metadata, dict):
                name = metadata.get("filename")
                if isinstance(name, str) and name:
                    filenames.add(name)
        if offset is None:
            break

    return JSONResponse(
        {
            "collection": collection,
            "category": category,
            "count": len(filenames),
            "filenames": sorted(filenames),
        },
        status_code=200,
    )

@app.post("/process")
async def process_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    collection: str = Form(...),
    job_key: Optional[str] = Form(None),
    # How to signal "already exists": conflict|ok|skip (default = conflict)
    exists_mode: str = Query("conflict", pattern="^(ok|conflict|skip)$"),
    # Single optional async toggle (default async=1 -> background ingestion)
    async_mode: int = Query(1, ge=0, le=1),
):
    filename = file.filename or f"upload_{uuid.uuid4().hex}"
    ext = os.path.splitext(filename)[1].lower()
    filename_no_ext = os.path.splitext(filename)[0]

    client = qdrant_client()

    # Deterministic ID retained only as a hint for clients expecting it.
    marker_id_hint = _marker_id_for_filename(collection, filename_no_ext)

    existing_marker_id: Optional[str] = None
    existing_reason: Optional[str] = None
    marker_status: Optional[str] = None

    # ---------- Pre-check: look for prior ingestion ----------
    try:
        _idempotent_create_indexes(
            client,
            collection,
            fields=(
                "metadata.category",
                "metadata.filename",
                "type",
            ),
        )
    except Exception:
        pass

    # Prefer completed job markers when they exist.
    marker_records: List[Any] = []
    try:
        marker_records, _ = client.scroll(
            collection_name=collection,
            scroll_filter={
                "must": [
                    {"key": "type", "match": {"value": "job_marker"}},
                    {"key": "metadata.filename", "match": {"value": filename_no_ext}},
                    {"key": "metadata.category", "match": {"value": category}},
                ]
            },
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        marker_records = []

    completed_marker = None
    for rec in marker_records:
        payload = rec.payload or {}
        metadata = payload.get("metadata") or {}
        rec_status = metadata.get("status")
        if rec_status == "done":
            completed_marker = rec
            marker_status = rec_status
            break

    if completed_marker:
        existing_reason = "existing_completed_marker"
        existing_marker_id = str(completed_marker.id)
    else:
        try:
            chunk_records, _ = client.scroll(
                collection_name=collection,
                scroll_filter={
                    "must": [
                        {"key": "metadata.filename", "match": {"value": filename_no_ext}},
                        {"key": "metadata.category", "match": {"value": category}},
                    ]
                },
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            chunk_records = []

        if chunk_records:
            existing_reason = "existing_chunks_same_filename"

    if existing_reason:  # already ingested -> signal based on exists_mode
        payload = {
            "status": "exists",
            "action": "conflict" if exists_mode == "conflict" else "skip",
            "file_name": filename_no_ext,
            "collection": collection,
            "reason": existing_reason,
            "marker_id": existing_marker_id or marker_id_hint,
            "category": category,
            "job_key": job_key,
        }
        if marker_status:
            payload["marker_status"] = marker_status
        if exists_mode == "conflict":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=payload)
        elif exists_mode == "skip":
            return JSONResponse({"status": "exists", "action": "skip", "file_name": filename_no_ext}, status_code=200)
        else:  # ok
            return JSONResponse(payload, status_code=200)

    # ---------- Not found: proceed ----------
    is_async = (async_mode == 1)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.file.seek(0)
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed_to_buffer_upload: {exc}",
        ) from exc
    file_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    if file_size == 0:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")

    job_key_final = job_key or uuid.uuid4().hex
    is_async = (async_mode == 1)

    if is_async:
        try:
            QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed_to_prepare_queue_directory: {exc}",
            ) from exc

        queued_basename = f"{job_key_final}{ext}"
        queued_path = QUEUE_DIR / queued_basename
        try:
            shutil.move(tmp_path, queued_path)
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed_to_enqueue_file: {exc}",
            ) from exc

        metadata = {
            "filename": filename,
            "category": category,
            "collection": collection,
            "job_key": job_key_final,
            "async_mode": 1,
            "file_size_bytes": file_size,
            "queued_file": str(queued_path),
        }
        try:
            meta_path = queued_path.with_suffix(queued_path.suffix + ".json")
            tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
            with tmp_meta_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_meta_path, meta_path)
        except Exception as exc:
            get_logger(job=job_key_final, file=filename, phase="queue").warning(
                f"Failed to write queue metadata: {exc}"
            )
            meta_path = None

        worker_lock = QUEUE_DIR / ".worker.lock"
        queue_status = "queued"
        if not worker_lock.exists():
            try:
                background_tasks.add_task(run_queue_once)
                queue_status = "scheduled"
            except Exception as exc:
                get_logger(job=job_key_final, file=filename, phase="queue").warning(
                    f"Failed to schedule queue worker: {exc}"
                )
        else:
            get_logger(job=job_key_final, file=filename, phase="queue").info(
                "Queue worker already active; job enqueued"
            )

        return JSONResponse(
            {
                "status": "accepted",
                "async_mode": 1,
                "file_name": filename_no_ext,
                "collection": collection,
                "category": category,
                "job_key": job_key_final,
                "file_size_bytes": file_size,
                "queued_file": str(queued_path),
                "metadata_file": str(meta_path) if meta_path else None,
                "queue_status": queue_status,
            },
            status_code=status.HTTP_202_ACCEPTED,
        )

    # Sync path
    try:
        result = process_file(
            file_path=tmp_path,
            filename=filename,
            ext=ext,
            category=category,
            collection=collection,
            job_key=job_key_final,
        )
        result["file_size_bytes"] = file_size
        return JSONResponse(result, status_code=200)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Return basic health information and configuration presence."""
    info = {
        "python": os.sys.version.split()[0],
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "embed_model": os.getenv("GENAI_EMBED_MODEL", "gemini-embedding-001"),
        "api_key_present": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")),
        "max_concurrent_vlm": os.getenv("MAX_CONCURRENT_VLM", "5"),
        "default_async": True,
        "exists_mode_default": "conflict",
    }
    return JSONResponse(content=info)


__all__ = ["app"]

