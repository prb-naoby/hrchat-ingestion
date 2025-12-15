from __future__ import annotations

import base64
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).strftime(ISO_FMT) + "Z"


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


@dataclass
class FigureState:
    index: int
    page: Optional[int]
    caption: str
    jpeg_b64: str
    description: str = ""
    completed: bool = False

    @classmethod
    def from_bytes(cls, index: int, page: Optional[int], caption: str, jpeg_bytes: bytes) -> "FigureState":
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        return cls(index=index, page=page, caption=caption, jpeg_b64=encoded, description="", completed=False)

    def to_bytes(self) -> bytes:
        return base64.b64decode(self.jpeg_b64.encode("ascii"))


class ParseProgressTracker:
    VERSION = 2
    DEFAULT_PAGE_CHECKPOINT = 25
    DEFAULT_HEARTBEAT_SECONDS = 300
    FIGURE_CHECKPOINT = 5

    def __init__(
        self,
        cache_path: str | Path,
        log,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        page_checkpoint: int = DEFAULT_PAGE_CHECKPOINT,
        heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self.path = Path(cache_path)
        self.log = log
        self.progress_callback = progress_callback
        self.page_checkpoint = max(1, page_checkpoint)
        self.heartbeat_seconds = max(30, heartbeat_seconds)

        self._data: Dict[str, Any] = self._load()
        self._last_flush_ts = 0.0
        self._last_heartbeat_ts = 0.0
        self._next_page_log = self.page_checkpoint
        self._next_fig_log = self.FIGURE_CHECKPOINT
        self._last_phase_snapshot: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": self.VERSION, "heartbeat": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": self.VERSION, "heartbeat": {}}

        if isinstance(raw, dict) and "version" not in raw and "markdown" in raw and "metadata" in raw:
            # Legacy cache (final only)
            return {"version": self.VERSION, "heartbeat": {}, "final": raw}

        if not isinstance(raw, dict):
            return {"version": self.VERSION, "heartbeat": {}}

        raw.setdefault("version", self.VERSION)
        raw.setdefault("heartbeat", {})
        return raw

    def _maybe_flush(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_flush_ts) < 2.0:
            return
        try:
            _write_json_atomic(self.path, self._data)
            self._last_flush_ts = now
        except Exception:
            pass

    def _heartbeat(
        self,
        phase: str,
        *,
        pages_processed: Optional[int] = None,
        pages_total: Optional[int] = None,
        figures_described: Optional[int] = None,
        figures_total: Optional[int] = None,
        batch_index: Optional[int] = None,
        batches_total: Optional[int] = None,
    ) -> None:
        now = time.time()
        if (now - self._last_heartbeat_ts) < 1.0 and not phase == "final":
            # Avoid writing dozens of updates per second
            return
        heartbeat = {
            "phase": phase,
            "heartbeat_ts": _utcnow_iso(),
        }
        if pages_processed is not None:
            heartbeat["pages_processed"] = pages_processed
        if pages_total is not None:
            heartbeat["pages_total"] = pages_total
        if figures_described is not None:
            heartbeat["figures_described"] = figures_described
        if figures_total is not None:
            heartbeat["figures_total"] = figures_total
        if batch_index is not None:
            heartbeat["batch_index"] = batch_index
        if batches_total is not None:
            heartbeat["batches_total"] = batches_total
        self._data["heartbeat"] = heartbeat
        self._last_heartbeat_ts = now
        if self.progress_callback:
            try:
                self.progress_callback(dict(heartbeat))
            except Exception:
                pass
        self._maybe_flush()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def has_final(self) -> bool:
        final = self._data.get("final")
        return isinstance(final, dict) and "markdown" in final and "metadata" in final

    def get_final(self) -> tuple[str, Dict[str, Any]]:
        final = self._data.get("final") or {}
        return final.get("markdown", ""), final.get("metadata", {})  # type: ignore[return-value]

    def doc_ready(self) -> bool:
        doc = self._data.get("doc")
        return isinstance(doc, dict) and "base_markdown" in doc and "figures" in doc

    def get_doc_state(self) -> Dict[str, Any]:
        doc = self._data.get("doc") or {}
        return doc  # type: ignore[return-value]

    def existing_descriptions(self) -> Dict[int, str]:
        doc = self._data.get("doc")
        if not isinstance(doc, dict):
            return {}
        figures = doc.get("figures")
        if not isinstance(figures, list):
            return {}
        results: Dict[int, str] = {}
        for entry in figures:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            desc = entry.get("description")
            completed = entry.get("completed")
            if isinstance(idx, int) and completed is True:
                results[idx] = desc if isinstance(desc, str) else ""
        return results

    def store_doc_output(
        self,
        *,
        base_markdown: str,
        metadata: Dict[str, Any],
        figures: List[FigureState],
        next_page: Optional[int] = None,
        pages_processed: Optional[int] = None,
        pages_total: Optional[int] = None,
        batch_index: Optional[int] = None,
        batches_total: Optional[int] = None,
    ) -> None:
        metadata_copy = copy.deepcopy(metadata)
        if pages_total is None:
            pages_total = metadata_copy.get("page_count")
        if pages_total is not None:
            metadata_copy["page_count"] = pages_total
        described_count = sum(1 for figure in figures if figure.description or figure.completed)
        doc = {
            "base_markdown": base_markdown,
            "metadata": metadata_copy,
            "figures": [figure.__dict__ for figure in figures],
            "figures_total": len(figures),
            "pages_total": pages_total,
            "stored_ts": _utcnow_iso(),
            "figures_described": described_count,
        }
        if next_page is not None:
            doc["next_page"] = next_page
        if batch_index is not None:
            doc["batch_index"] = batch_index
        if batches_total is not None:
            doc["batches_total"] = batches_total
        self._data["doc"] = doc
        self._next_page_log = self.page_checkpoint
        self._next_fig_log = self.FIGURE_CHECKPOINT
        self.update_phase(
            "docling",
            pages_processed=pages_processed,
            pages_total=pages_total,
            figures_described=described_count,
            figures_total=len(figures),
            batch_index=batch_index,
            batches_total=batches_total,
        )
        self._maybe_flush(force=True)

    def resume(self) -> None:
        doc = self._data.get("doc")
        if not isinstance(doc, dict):
            return
        figures = doc.get("figures")
        described = 0
        if isinstance(figures, list):
            for entry in figures:
                if isinstance(entry, dict) and isinstance(entry.get("description"), str) and entry["description"]:
                    described += 1
                elif isinstance(entry, dict) and entry.get("completed") is True:
                    described += 1
        total = self.total_figures()
        pages_total = doc.get("pages_total")
        next_page = doc.get("next_page")
        batch_index = doc.get("batch_index")
        batches_total = doc.get("batches_total")
        if isinstance(pages_total, int):
            if isinstance(next_page, int):
                pages_processed = min(max(next_page - 1, 0), pages_total)
            else:
                pages_processed = pages_total
        else:
            pages_processed = None
        doc["figures_described"] = described
        pending_pages = (
            (pages_total - max(next_page - 1, 0))
            if isinstance(pages_total, int) and isinstance(next_page, int)
            else 0
        )
        if isinstance(pending_pages, int) and pending_pages > 0:
            phase = "docling"
        elif described and described < total:
            phase = "figures"
        else:
            phase = "docling"
        if described == total and total and pending_pages <= 0:
            phase = "compose"
        self._heartbeat(
            phase,
            pages_processed=pages_processed,
            pages_total=pages_total if isinstance(pages_total, int) else None,
            figures_described=described,
            figures_total=total,
            batch_index=batch_index if isinstance(batch_index, int) else None,
            batches_total=batches_total if isinstance(batches_total, int) else None,
        )

    def log_doc_progress(self, pages_processed: int, pages_total: Optional[int]) -> None:
        if pages_processed >= self._next_page_log or (pages_total and pages_processed >= pages_total):
            if pages_total:
                self.log.info(f"Parse progress: pages={pages_processed}/{pages_total}")
            else:
                self.log.info(f"Parse progress: pages={pages_processed}")
            self._next_page_log = pages_processed + self.page_checkpoint
        self._heartbeat("docling", pages_processed=pages_processed, pages_total=pages_total, figures_total=self.total_figures())

    def update_phase(
        self,
        phase: str,
        *,
        pages_processed: Optional[int] = None,
        pages_total: Optional[int] = None,
        figures_described: Optional[int] = None,
        figures_total: Optional[int] = None,
        batch_index: Optional[int] = None,
        batches_total: Optional[int] = None,
    ) -> None:
        snapshot = {
            "phase": phase,
            "pages_processed": pages_processed,
            "pages_total": pages_total,
            "figures_described": figures_described,
            "figures_total": figures_total,
            "batch_index": batch_index,
            "batches_total": batches_total,
        }
        if snapshot != self._last_phase_snapshot:
            parts = [f"phase={phase}"]
            if pages_processed is not None:
                total_str = str(pages_total) if pages_total is not None else "?"
                parts.append(f"pages={pages_processed}/{total_str}")
            if figures_described is not None:
                fig_total = figures_total if figures_total is not None else "?"
                parts.append(f"figures={figures_described}/{fig_total}")
            if batch_index is not None:
                batch_total_str = str(batches_total) if batches_total is not None else "?"
                parts.append(f"batch={batch_index}/{batch_total_str}")
            self.log.info("Parse heartbeat: " + " ".join(parts))
            self._last_phase_snapshot = snapshot
        self._heartbeat(
            phase,
            pages_processed=pages_processed,
            pages_total=pages_total,
            figures_described=figures_described,
            figures_total=figures_total,
            batch_index=batch_index,
            batches_total=batches_total,
        )

    def total_figures(self) -> int:
        doc = self._data.get("doc")
        if isinstance(doc, dict):
            val = doc.get("figures_total")
            if isinstance(val, int):
                return val
            figures = doc.get("figures")
            if isinstance(figures, list):
                return len(figures)
        return 0

    def record_description(self, index: int, description: str) -> None:
        doc = self._data.get("doc")
        if not isinstance(doc, dict):
            return
        figures = doc.get("figures")
        if not isinstance(figures, list):
            return
        described = 0
        for entry in figures:
            if not isinstance(entry, dict):
                continue
            if entry.get("index") == index:
                entry["description"] = description
                entry["completed"] = True
            if isinstance(entry.get("description"), str) and entry["description"]:
                described += 1
            elif entry.get("completed") is True:
                described += 1
        doc["figures_described"] = described
        if described >= self._next_fig_log or described >= len(figures):
            self.log.info(
                f"Parse progress: figures={described}/{len(figures)}"
            )
            self._next_fig_log = described + self.FIGURE_CHECKPOINT
        pages_total = doc.get("pages_total")
        next_page = doc.get("next_page")
        if isinstance(pages_total, int):
            if isinstance(next_page, int):
                pages_processed = min(max(next_page - 1, 0), pages_total)
            else:
                pages_processed = pages_total
        else:
            pages_processed = None
        batch_index = doc.get("batch_index")
        batches_total = doc.get("batches_total")
        self._heartbeat(
            "figures",
            pages_processed=pages_processed,
            pages_total=pages_total if isinstance(pages_total, int) else None,
            figures_described=described,
            figures_total=len(figures),
            batch_index=batch_index if isinstance(batch_index, int) else None,
            batches_total=batches_total if isinstance(batches_total, int) else None,
        )
        self._maybe_flush()

    def finalize(self, markdown: str, metadata: Dict[str, Any]) -> None:
        self._data["final"] = {"markdown": markdown, "metadata": metadata, "completed_ts": _utcnow_iso()}
        figures_total = self.total_figures()
        described = None
        doc = self._data.get("doc")
        if isinstance(doc, dict):
            described = doc.get("figures_described")
            if not isinstance(described, int):
                figures = doc.get("figures")
                if isinstance(figures, list):
                    described = sum(1 for entry in figures if isinstance(entry, dict) and entry.get("completed") is True)
        pages_total = metadata.get("page_count") or (doc.get("pages_total") if isinstance(doc, dict) else None)
        pages_processed = pages_total if isinstance(pages_total, int) else None
        batch_index = doc.get("batch_index") if isinstance(doc, dict) else None
        batches_total = doc.get("batches_total") if isinstance(doc, dict) else None
        self._heartbeat(
            "final",
            pages_processed=pages_processed,
            pages_total=pages_total if isinstance(pages_total, int) else None,
            figures_described=described,
            figures_total=figures_total,
            batch_index=batch_index if isinstance(batch_index, int) else None,
            batches_total=batches_total if isinstance(batches_total, int) else None,
        )
        self._maybe_flush(force=True)
