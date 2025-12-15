"""Queued ingestion worker.

This module scans `INGEST_QUEUE_DIR` for enqueued documents (written by the
FastAPI layer), optionally retries unfinished jobs from `temp/`, and runs the
standard synchronous pipeline after performing duplicate checks.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime, timedelta

from .pipeline import process_file
from .qdrant_utils import qdrant_client
from .logger import get_logger

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

QUEUE_DIR = Path(os.getenv("INGEST_QUEUE_DIR", "ingest_queue"))
TEMP_DIR = Path(os.getenv("INGEST_TEMP_DIR", "temp"))
WORKER_CONCURRENCY = max(1, int(os.getenv("INGEST_WORKER_CONCURRENCY", "1")))
LOCK_TTL_SECONDS = float(os.getenv("INGEST_LOCK_TTL_SEC", "300"))
MAX_RETRIES = max(1, int(os.getenv("INGEST_MAX_RETRIES", "3")))
FAILED_COOLDOWN_DAYS = int(os.getenv("INGEST_FAILED_COOLDOWN_DAYS", "7"))
FAILED_INDEX_PATH = QUEUE_DIR / "failed_jobs.json"


def _load_metadata(meta_path: Path) -> Dict[str, Any]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["_meta_path"] = str(meta_path)
    return data


def _load_failed_index() -> Dict[str, Any]:
    if FAILED_INDEX_PATH.exists():
        try:
            return json.loads(FAILED_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"jobs": {}, "filenames": {}}
    return {"jobs": {}, "filenames": {}}


_FAILED_INDEX: Dict[str, Any] = _load_failed_index()


def _persist_failed_index() -> None:
    try:
        FAILED_INDEX_PATH.write_text(
            json.dumps(_FAILED_INDEX, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _record_failed_job(
    queue_pdf: Path, metadata: Dict[str, Any], reason: str, *, drop: bool = False
) -> None:
    job_key = queue_pdf.stem
    filename = metadata.get("filename") or queue_pdf.name
    cooldown = datetime.utcnow() + timedelta(days=FAILED_COOLDOWN_DAYS)
    entry = {
        "filename": filename,
        "reason": reason,
        "dropped": drop,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cooldown_until": cooldown.isoformat(timespec="seconds") + "Z",
    }
    progress = metadata.get("parse_progress")
    if isinstance(progress, dict):
        entry["parse_progress"] = dict(progress)
    cache_path = metadata.get("cache_path")
    if drop and cache_path:
        try:
            Path(cache_path).unlink(missing_ok=True)
        except Exception:
            pass
    _FAILED_INDEX.setdefault("jobs", {})[job_key] = entry
    if filename:
        _FAILED_INDEX.setdefault("filenames", {})[filename.lower()] = entry
    _persist_failed_index()


def _remove_failed_entry(job_key: str, filename: str | None = None) -> None:
    if job_key in _FAILED_INDEX.get("jobs", {}):
        _FAILED_INDEX["jobs"].pop(job_key, None)
    if filename:
        _FAILED_INDEX.get("filenames", {}).pop(filename.lower(), None)
    _persist_failed_index()


def _get_failed_entry(queue_pdf: Path, metadata: Dict[str, Any]) -> Dict[str, Any] | None:
    job_key = queue_pdf.stem
    filename = (metadata.get("filename") or queue_pdf.name or "").lower()
    entry = _FAILED_INDEX.get("jobs", {}).get(job_key)
    if not entry and filename:
        entry = _FAILED_INDEX.get("filenames", {}).get(filename)
    if not entry:
        return None

    cooldown_until = entry.get("cooldown_until")
    if cooldown_until:
        try:
            dt = datetime.fromisoformat(cooldown_until.replace("Z", "+00:00"))
        except Exception:
            dt = None
        if dt and datetime.utcnow() > dt:
            _remove_failed_entry(job_key, filename if filename else None)
            return None
    return entry


def _write_metadata(meta_path: Path, metadata: Dict[str, Any]) -> None:
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, meta_path)


def _requeue_or_drop(
    queue_pdf: Path,
    processing_pdf: Path,
    meta_path: Path,
    metadata: Dict[str, Any],
    attempts: int,
    reason: str,
    log,
) -> int:
    next_attempt = attempts + 1
    metadata["last_error"] = reason
    metadata["last_error_ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if next_attempt >= MAX_RETRIES:
        log.error(f"Dropping job after {next_attempt} attempts: {reason}")
        _record_failed_job(queue_pdf, metadata, reason, drop=True)
        processing_pdf.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return 0

    metadata["attempts"] = next_attempt
    try:
        _write_metadata(meta_path, metadata)
    except Exception as exc:
        log.warning(f"Failed to update metadata for retry: {exc}")
    try:
        processing_pdf.rename(queue_pdf)
    except Exception as exc:
        log.error(f"Failed to requeue job: {exc}")
        _record_failed_job(queue_pdf, metadata, f"requeue-failed: {exc}", drop=True)
        return 0
    log.info(f"Job requeued for retry attempt {next_attempt}/{MAX_RETRIES}")
    return 0


def _existing_in_qdrant(collection: str, filename: str, category: str) -> bool:
    client = qdrant_client()
    try:
        records, _ = client.scroll(
            collection_name=collection,
            scroll_filter={
                "must": [
                    {"key": "metadata.filename", "match": {"value": filename}},
                    {"key": "metadata.category", "match": {"value": category}},
                ]
            },
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except Exception:
        records = []
    return bool(records)


def _collect_temp_jobs() -> Dict[str, Tuple[Path, Path]]:
    pending: Dict[str, Tuple[Path, Path]] = {}
    if not TEMP_DIR.exists():
        return pending
    for meta_path in TEMP_DIR.glob("*.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_key = data.get("job_key")
        file_path = data.get("queued_file") or data.get("file_path")
        if not job_key or not file_path:
            continue
        pdf_path = Path(file_path)
        if pdf_path.exists():
            pending[job_key] = (pdf_path, meta_path)
    return pending


def _enqueue_from_temp(log):
    pending = _collect_temp_jobs()
    for job_key, (pdf_path, meta_path) in pending.items():
        log.info(f"Recovering unfinished job {job_key} from temp directory")
        queue_pdf = QUEUE_DIR / pdf_path.name
        queue_meta = queue_pdf.with_suffix(queue_pdf.suffix + ".json")
        try:
            QUEUE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(pdf_path, queue_pdf)
            shutil.move(meta_path, queue_meta)
        except Exception as exc:
            log.warning(f"Failed to move temp job {job_key} back to queue: {exc}")


def _restore_processing_files(log) -> None:
    """Return any orphaned *.processing files back to queue."""
    for processing_path in QUEUE_DIR.glob("*.processing"):
        target = processing_path.with_suffix("")
        try:
            if target.exists():
                processing_path.unlink()
            else:
                processing_path.rename(target)
        except Exception as exc:
            log.warning(f"Failed to restore {processing_path.name}: {exc}")


def _acquire_lock(log) -> Path | None:
    lock_path = QUEUE_DIR / ".worker.lock"
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > LOCK_TTL_SECONDS:
                log.warning(f"Recovering stale queue lock (age={age:.1f}s)")
                lock_path.unlink(missing_ok=True)
            else:
                return None
        except Exception:
            return None
    try:
        fh = lock_path.open("x")
        fh.write(str(os.getpid()))
        fh.close()
        return lock_path
    except Exception:
        return None


def _claim_job(pdf_path: Path) -> Path | None:
    try:
        processing_path = pdf_path.with_suffix(pdf_path.suffix + ".processing")
        pdf_path.rename(processing_path)
        return processing_path
    except FileNotFoundError:
        return None
    except PermissionError:
        return None
    except Exception:
        return None


def _process_job(pdf_processing_path: Path, meta_path: Path) -> int:
    queue_pdf = pdf_processing_path.with_suffix("")
    log = get_logger(job=pdf_processing_path.stem, file=meta_path.stem, phase="queue")
    metadata = _load_metadata(meta_path)
    job_key = metadata.get("job_key") or pdf_processing_path.stem
    cache_dir = QUEUE_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = metadata.get("cache_path")
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        cache_file = cache_dir / f"{job_key}.json"
        metadata["cache_path"] = str(cache_file)
        metadata["cache_created_ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        _write_metadata(meta_path, metadata)
    filename = metadata.get("filename") or queue_pdf.name
    collection = metadata.get("collection") or os.getenv("QDRANT_COLLECTION", "documents")
    category = metadata.get("category") or os.getenv("DOC_CATEGORY", "general")
    attempts = int(metadata.get("attempts", 0))
    def _progress_callback(update: Dict[str, Any]) -> None:
        progress_copy = dict(update) if isinstance(update, dict) else {}
        metadata["parse_progress"] = progress_copy
        heartbeat_ts = progress_copy.get("heartbeat_ts")
        if isinstance(heartbeat_ts, str):
            metadata["parse_progress_ts"] = heartbeat_ts
        phase = progress_copy.get("phase")
        pages_processed = progress_copy.get("pages_processed")
        pages_total = progress_copy.get("pages_total")
        figures_described = progress_copy.get("figures_described")
        figures_total = progress_copy.get("figures_total")
        batch_index = progress_copy.get("batch_index")
        batches_total = progress_copy.get("batches_total")
        pct = None
        if isinstance(pages_processed, (int, float)) and isinstance(pages_total, (int, float)) and pages_total:
            pct = round(float(pages_processed) / float(pages_total) * 100, 2)
        try:
            _write_metadata(meta_path, metadata)
        except Exception:
            pass
        parts = [f"phase={phase}"] if phase else ["phase=unknown"]
        if isinstance(pages_processed, (int, float)):
            total_str = str(pages_total) if isinstance(pages_total, (int, float)) else "?"
            parts.append(f"pages={pages_processed}/{total_str}")
        if isinstance(figures_described, (int, float)):
            fig_total = str(figures_total) if isinstance(figures_total, (int, float)) else "?"
            parts.append(f"figures={figures_described}/{fig_total}")
        if isinstance(batch_index, (int, float)):
            batch_total = str(batches_total) if isinstance(batches_total, (int, float)) else "?"
            parts.append(f"batch={batch_index}/{batch_total}")
        if pct is not None:
            parts.append(f"progress={pct:.2f}%")
        log.info("Queue progress: " + " ".join(parts))

    progress_entry = metadata.get("parse_progress")
    if isinstance(progress_entry, dict):
        hb_ts = progress_entry.get("heartbeat_ts")
        if isinstance(hb_ts, str):
            try:
                hb_dt = datetime.fromisoformat(hb_ts.replace("Z", "+00:00"))
                age = (datetime.utcnow() - hb_dt).total_seconds()
                if age > 900:
                    log.info(f"Detected stale parse heartbeat age={age:.0f}s; resuming from cache")
            except Exception:
                pass

    base_name = Path(filename).stem
    if _existing_in_qdrant(collection, base_name, category):
        log.info(f"Skipping {filename}: already present in Qdrant")
        pdf_processing_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return 0

    ext = Path(filename).suffix.lower()
    if not ext:
        suffixes = pdf_processing_path.suffixes
        if len(suffixes) >= 2:
            ext = suffixes[-2].lower()

    log.info(
        "Starting ingestion "
        f"(size_bytes={metadata.get('file_size_bytes')}, pages={metadata.get('page_count')})"
    )

    try:
        result = process_file(
            file_path=str(pdf_processing_path),
            cache_path=str(cache_file),
            filename=filename,
            ext=ext,
            category=category,
            collection=collection,
            job_key=job_key,
            progress_callback=_progress_callback,
        )
    except Exception as exc:
        log.error(f"Ingestion failed: {exc}")
        return _requeue_or_drop(queue_pdf, pdf_processing_path, meta_path, metadata, attempts, str(exc), log)
    else:
        uploaded = result.get("uploaded_chunks")
        if not uploaded:
            reason = f"uploaded_chunks={uploaded}"
            return _requeue_or_drop(queue_pdf, pdf_processing_path, meta_path, metadata, attempts, reason, log)
        _remove_failed_entry(job_key, filename)
        log.info(f"Ingestion completed: uploaded_chunks={uploaded}")
        pdf_processing_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return 1


def run_once() -> int:
    log = get_logger(phase="queue")
    lock_path = _acquire_lock(log)
    if not lock_path:
        return 0

    processed = 0
    try:
        _enqueue_from_temp(log)
        _restore_processing_files(log)

        if not QUEUE_DIR.exists():
            log.info("Queue directory not found; nothing to do")
            return 0

        with ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY) as executor:
            futures = set()

            def wait_for_completion(block: bool) -> None:
                nonlocal processed
                done = [f for f in list(futures) if f.done()]
                if not done and block and futures:
                    done = [next(as_completed(futures))]
                for fut in done:
                    futures.discard(fut)
                    try:
                        processed += fut.result() or 0
                    except Exception as exc:
                        log.error(f"Queue worker job failed: {exc}")

            while True:
                jobs = sorted(QUEUE_DIR.glob("*.pdf"))
                for pdf_path in jobs:
                    if len(futures) >= WORKER_CONCURRENCY:
                        break
                    meta_path = pdf_path.with_suffix(pdf_path.suffix + ".json")
                    if not meta_path.exists():
                        log.warning(f"Missing metadata sidecar for {pdf_path.name}; skipping")
                        continue
                    metadata = _load_metadata(meta_path)
                    failed_entry = _get_failed_entry(pdf_path, metadata)
                    if failed_entry:
                        log.info(
                            "Skipping failed document "
                            f"{metadata.get('filename') or pdf_path.name}: {failed_entry.get('reason')} "
                            f"(cooldown_until={failed_entry.get('cooldown_until')})"
                        )
                        cache_path = metadata.get("cache_path")
                        if cache_path:
                            try:
                                Path(cache_path).unlink(missing_ok=True)
                            except Exception:
                                pass
                        pdf_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        continue
                    claimed = _claim_job(pdf_path)
                    if not claimed:
                        continue
                    futures.add(executor.submit(_process_job, claimed, meta_path))
                if not futures:
                    if not jobs:
                        break
                wait_for_completion(block=bool(futures))
            wait_for_completion(block=False)
        return processed
    finally:
        lock_path.unlink(missing_ok=True)


def run_forever(poll_interval: float = 10.0) -> None:
    log = get_logger(phase="queue")
    log.info("Starting queue worker loop")
    while True:
        count = run_once()
        if count == 0:
            time.sleep(poll_interval)
        else:
            continue


def get_failed_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    stale: list[tuple[str, str | None]] = []
    for job_key, entry in list(_FAILED_INDEX.get("jobs", {}).items()):
        cooldown_until = entry.get("cooldown_until")
        filename = entry.get("filename")
        if cooldown_until:
            try:
                dt = datetime.fromisoformat(cooldown_until.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt and datetime.utcnow() > dt:
                stale.append((job_key, filename))
                continue
        jobs.append({"job_key": job_key, **entry})
    for job_key, filename in stale:
        _remove_failed_entry(job_key, filename)
    return jobs


def get_queue_status() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    meta_paths = sorted(QUEUE_DIR.glob("*.pdf.json"))
    for meta_path in meta_paths:
        try:
            metadata = _load_metadata(meta_path)
        except Exception:
            continue
        queue_pdf = Path(meta_path.with_suffix(""))
        processing_path = queue_pdf.with_suffix(queue_pdf.suffix + ".processing")
        if processing_path.exists():
            status = "processing"
            stat_path = processing_path
        elif queue_pdf.exists():
            status = "queued"
            stat_path = queue_pdf
        else:
            status = "pending"
            stat_path = meta_path
        parse_progress = metadata.get("parse_progress")
        if not isinstance(parse_progress, dict):
            parse_progress = {}
        heartbeat_ts = parse_progress.get("heartbeat_ts")
        heartbeat_age = None
        if isinstance(heartbeat_ts, str):
            try:
                hb_dt = datetime.fromisoformat(heartbeat_ts.replace("Z", "+00:00"))
                heartbeat_age = (datetime.utcnow() - hb_dt).total_seconds()
            except Exception:
                heartbeat_age = None
        pages_processed = parse_progress.get("pages_processed")
        pages_total = parse_progress.get("pages_total")
        progress_pct: Optional[float] = None
        if isinstance(pages_processed, (int, float)) and isinstance(pages_total, (int, float)) and pages_total:
            progress_pct = round(float(pages_processed) / float(pages_total) * 100.0, 2)
        job_key = metadata.get("job_key") or queue_pdf.stem.replace(".pdf", "")
        job_entry: Dict[str, Any] = {
            "job_key": job_key,
            "filename": metadata.get("filename"),
            "category": metadata.get("category"),
            "status": status,
            "file_size_bytes": metadata.get("file_size_bytes"),
            "attempts": metadata.get("attempts", 0),
            "parse_progress": parse_progress,
            "progress_percent": progress_pct,
            "heartbeat_age_seconds": heartbeat_age,
            "metadata_path": str(meta_path),
            "queued_file": str(queue_pdf) if queue_pdf.exists() else None,
            "parse_progress_ts": heartbeat_ts if isinstance(heartbeat_ts, str) else None,
            "cache_path": metadata.get("cache_path"),
            "cache_created_ts": metadata.get("cache_created_ts"),
        }
        batch_index = parse_progress.get("batch_index")
        batches_total = parse_progress.get("batches_total")
        if isinstance(batch_index, (int, float)):
            job_entry["batch_index"] = batch_index
        if isinstance(batches_total, (int, float)):
            job_entry["batches_total"] = batches_total
        try:
            stat_mtime = stat_path.stat().st_mtime
            job_entry["updated_ts"] = datetime.utcfromtimestamp(stat_mtime).isoformat(timespec="seconds") + "Z"
        except Exception:
            job_entry["updated_ts"] = None
        jobs.append(job_entry)
    jobs.sort(
        key=lambda j: (
            0 if j.get("status") == "processing" else 1,
            j.get("parse_progress_ts") or "",
            j.get("filename") or "",
        )
    )
    return jobs


__all__ = ["run_once", "run_forever", "get_failed_jobs", "get_queue_status"]















