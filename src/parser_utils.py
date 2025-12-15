"""Parsing and figure description utilities.

This module wraps the heavy docling parsing functionality in a synchronous API
and adds concurrency around calls to the Gemini visual language model
(VLM).  It is responsible for turning raw documents (PDFs or images)
into enriched markdown and extracting useful metadata.

The primary entry point is :func:`parse_document`, which handles both
PDF and image inputs.  Internally it calls either :func:`_parse_pdf`
or :func:`_parse_image`.
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from .doc_parser import DocParser, Figure  # parses PDFs and extracts figures
from .markdown import FigureDesc, compose_markdown
from .vlm import GeminiVLM, VLMConfig
from .prompts import default_prompt, ocr_prompt
from .util_img import to_jpeg_bytes
from .logger import get_logger, MAX_CONCURRENT_VLM
from .parse_cache import FigureState, ParseProgressTracker

try:  # Optional dependency for fallback page rendering
    import fitz  # type: ignore

    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover - optional path
    fitz = None  # type: ignore
    _HAS_PYMUPDF = False

OCR_MIN_CHARS_PER_PAGE = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "200"))
OCR_MIN_LINES_PER_PAGE = int(os.getenv("OCR_MIN_LINES_PER_PAGE", "3"))
OCR_RENDER_DPI = int(os.getenv("OCR_RENDER_DPI", "220"))
OCR_RENDER_MAX_PAGES = int(os.getenv("OCR_RENDER_MAX_PAGES", "500"))
VLM_FIGURE_TIMEOUT = float(os.getenv("VLM_FIGURE_TIMEOUT_SEC", "60"))
PARSE_PAGE_BATCH_SIZE = max(1, int(os.getenv("PARSE_PAGE_BATCH_SIZE", "50")))
PARSE_PAGE_BATCH_PERCENT = float(os.getenv("PARSE_PAGE_BATCH_PERCENT", "5"))


def _pdf_page_count(path: str) -> Optional[int]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(path)
        return len(reader.pages)
    except Exception:
        return None


def _ensure_primitive(value: Any) -> Any:
    """Coerce arbitrary objects into JSON-serialisable primitives.

    If the value is callable it will be invoked (best effort).  Containers are
    traversed recursively.  Unserialisable objects are converted to strings.
    """
    if callable(value):
        try:
            value = value()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {k: _ensure_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_primitive(v) for v in value]
    try:
        import json  # local import to avoid circular deps
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def extract_metadata_from_doc(doc: Any) -> Dict[str, Any]:
    """Extract common metadata from a Docling document if present.

    Attempts to read author(s) and number of pages.  When information is
    unavailable it returns ``None`` for those fields.
    """
    author: Optional[str] = None
    page_count: Optional[int] = None
    try:
        if hasattr(doc, "info"):
            info = getattr(doc, "info")
            # First try a plural authors field
            if hasattr(info, "authors"):
                try:
                    authors_val = getattr(info, "authors")
                    authors = _ensure_primitive(authors_val)
                    if authors:
                        if isinstance(authors, (list, tuple, set)):
                            author = ", ".join(str(a) for a in authors)
                        else:
                            author = str(authors)
                except Exception:
                    pass
            # Fallback to singular author field
            if author is None and hasattr(info, "author"):
                try:
                    val = getattr(info, "author")
                    primitive = _ensure_primitive(val)
                    if primitive:
                        author = str(primitive)
                except Exception:
                    pass
        for attr in ["page_count", "n_pages", "num_pages"]:
            if page_count is not None:
                break
            if hasattr(doc, attr):
                try:
                    val = getattr(doc, attr)
                    primitive = _ensure_primitive(val)
                    if primitive is not None:
                        try:
                            page_count = int(primitive)  # type: ignore[assignment]
                        except Exception:
                            page_count = None
                except Exception:
                    pass
    except Exception:
        pass
    return {"author": author, "page_count": page_count}


def clean_markdown(md: str) -> str:
    """Normalise markdown by collapsing excessive blank lines and stripping whitespace.

    This helper removes carriage returns, collapses three or more consecutive
    newlines into two and strips trailing spaces at the end of lines.  It
    improves the quality of the downstream chunks and embeddings.
    """
    # Normalise line endings
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse more than two blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Strip trailing spaces
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md.strip()


def _describe_one(vlm: GeminiVLM, f: Figure, prompt: str, job_key: str, filename: str) -> FigureDesc:
    """Describe a single figure with retries.

    Small images (less than ~5?kB) are assumed to be logos or decorative
    elements and skipped.  Descriptions shorter than ten characters are
    retried up to three times.  Any exceptions are caught and logged.
    """
    log = get_logger(job=job_key, file=filename, phase="vlm")
    if len(f.jpeg_bytes) < 5000:
        log.info(f"Skipping small figure {f.index} (page {f.page})")
        return FigureDesc(index=f.index, page=f.page, caption=f.caption, description="")
    text = ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            text = vlm.describe(f.jpeg_bytes, prompt)
            if text and len(text.strip()) >= 10:
                break
        except Exception as exc:
            log.warning(f"VLM describe failed for figure {f.index}: {exc}")
        time.sleep(0.5)
    if not text or len(text.strip()) < 10:
        log.warning(
            f"Failed to get a satisfactory description for figure {f.index} after {max_retries} attempts"
        )
        description = ""
    else:
        description = text.strip()
    return FigureDesc(index=f.index, page=f.page, caption=f.caption, description=description)


def _describe_figures(
    figures: List[Figure],
    prompt: str,
    job_key: str,
    filename: str,
    *,
    cache_tracker: Optional[ParseProgressTracker] = None,
) -> List[FigureDesc]:
    """Describe all figures concurrently using a thread pool.

    Respects the ``MAX_CONCURRENT_VLM`` limit.  Returns only the
    descriptions that are non-empty.  Entries with empty descriptions
    indicate that the figure was skipped.
    """
    if not figures:
        return []
    log = get_logger(job=job_key, file=filename, phase="vlm")
    log.info(f"Starting figure descriptions total={len(figures)}")
    start = time.perf_counter()
    vlm = GeminiVLM(VLMConfig())
    descs: List[FigureDesc] = []
    completed = 0
    described = 0
    half_threshold = max(1, len(figures) // 2)
    half_logged = False

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM) as executor:
        future_map = {
            executor.submit(_describe_one, vlm, f, prompt, job_key, filename): f for f in figures
        }
        for fut in as_completed(future_map):
            try:
                desc = fut.result()
                if desc.description:
                    descs.append(desc)
                    described += 1
                else:
                    descs.append(desc)
                completed += 1
                if cache_tracker:
                    cache_tracker.record_description(desc.index, desc.description or "")
                if not half_logged and completed >= half_threshold:
                    log.info(
                        f"Figure description progress completed={completed}/{len(future_map)} "
                        f"described={described}"
                    )
                    half_logged = True
            except Exception as exc:
                # Log unexpected errors and skip
                log.warning(f"Unexpected error describing figure: {exc}")
    if not half_logged and len(future_map) > 0:
        log.info(
            f"Figure description progress completed={completed}/{len(future_map)} "
            f"described={described}"
        )
    # Sort descriptions by index so they appear in document order
    descs.sort(key=lambda d: d.index)
    log.info(
        f"Finished figure descriptions described={described} skipped={len(figures) - described} "
        f"elapsed={time.perf_counter() - start:.2f}s"
    )
    return descs


def _calc_text_density(markdown: str) -> Tuple[int, int]:
    """Return (char_count, non_empty_line_count) for the supplied markdown."""
    stripped = markdown.strip()
    if not stripped:
        return 0, 0
    lines = stripped.splitlines()
    non_empty = sum(1 for line in lines if line.strip())
    return len(stripped), non_empty


def _should_trigger_ocr(char_count: int, line_count: int, page_count: Optional[int]) -> bool:
    """Decide whether OCR fallback should run based on text density."""
    if not page_count or page_count <= 0:
        return False
    if char_count == 0:
        return True
    chars_per_page = char_count / page_count
    lines_per_page = line_count / page_count
    if chars_per_page < OCR_MIN_CHARS_PER_PAGE:
        return True
    if lines_per_page < OCR_MIN_LINES_PER_PAGE:
        return True
    return False


def _render_pdf_pages_fallback(pdf_path: str, *, dpi: int, max_pages: int, log) -> List[Tuple[int, bytes]]:
    """Render PDF pages to JPEG bytes using PyMuPDF when Docling provides none."""
    if not _HAS_PYMUPDF:
        log.warning("Doc parser: PyMuPDF not available; cannot render fallback page images")
        return []

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - optional path
        log.warning(f"Doc parser: PIL unavailable for fallback rendering: {exc}")
        return []

    renders: List[Tuple[int, bytes]] = []
    page_limit = max_pages if max_pages > 0 else 0
    try:
        doc = fitz.open(pdf_path)  # type: ignore
        try:
            for page_index, page in enumerate(doc, start=1):
                if page_limit and page_index > page_limit:
                    break
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                jpeg_bytes = to_jpeg_bytes(img)
                renders.append((page_index, jpeg_bytes))
        finally:
            doc.close()
    except Exception as exc:
        log.warning(f"Doc parser: fallback page render failed: {exc}")
        renders.clear()
    return renders


def _ocr_document_pages(
    pages: List[Tuple[int, bytes]],
    job_key: str,
    filename: str,
) -> Tuple[str, Dict[str, Any]]:
    """Run Gemini OCR over per-page images and return markdown + metadata."""
    if not pages:
        return "", {"ocr_fallback": True, "ocr_pages": 0}
    vlm = GeminiVLM(VLMConfig())
    base_prompt = ocr_prompt()
    log = get_logger(job=job_key, file=filename, phase="ocr")
    sections: List[str] = []
    t_all = time.perf_counter()
    for page_no, image_bytes in sorted(pages, key=lambda x: x[0]):
        log.info(f"OCR fallback: transcribing page {page_no}")
        t_page = time.perf_counter()
        text = ""
        try:
            page_prompt = f"{base_prompt}\nHalaman ke-{page_no}. Salin teks apa adanya."
            text = vlm.transcribe(image_bytes, page_prompt)
        except Exception as exc:
            log.warning(f"OCR fallback: failed to transcribe page {page_no}: {exc}")
        elapsed = time.perf_counter() - t_page
        log.info(f"OCR fallback: page {page_no} completed in {elapsed:.2f}s")
        content = text.strip()
        if not content:
            continue
        sections.append(f"## Halaman {page_no}\n\n{content}")
    overall = time.perf_counter() - t_all
    log.info(f"OCR fallback: completed {len(sections)} pages in {overall:.2f}s")
    markdown = "\n\n".join(sections).strip()
    metadata = {
        "ocr_fallback": True,
        "ocr_pages": len(sections),
        "ocr_total_seconds": overall,
    }
    return markdown, metadata



def _parse_pdf(
    tmp_path: str,
    filename: str,
    job_key: str,
    *,
    cache_tracker: Optional[ParseProgressTracker] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Parse a PDF at ``tmp_path`` and return enriched markdown and metadata.

    The PDF is converted to markdown and figures using Docling. Figures are
    described concurrently via Gemini. The returned markdown includes a
    "Deskripsi Gambar" section appended to the end.
    """
    log = get_logger(job=job_key, file=filename, phase="parse")
    metadata: Dict[str, Any] = {"filename": filename, "file_type": "pdf"}
    parser = DocParser(keep_page_images=True)
    total_pages = _pdf_page_count(tmp_path)
    batch_size = PARSE_PAGE_BATCH_SIZE
    total_batches: Optional[int] = None
    overall_start = time.perf_counter()

    aggregated_base_md = ""
    aggregated_figures: List[Figure] = []
    aggregated_page_images: List[Tuple[int, bytes]] = []
    doc_loaded_from_cache = False
    next_page = 1
    image_only_doc = False
    aggregated_descs: Dict[int, FigureDesc] = {}
    existing_fig_descs: Dict[int, str] = cache_tracker.existing_descriptions() if cache_tracker else {}
    prompt = default_prompt()

    if cache_tracker and cache_tracker.doc_ready():
        doc_state = cache_tracker.get_doc_state()
        aggregated_base_md = doc_state.get("base_markdown", "") if isinstance(doc_state, dict) else ""
        if isinstance(doc_state, dict):
            doc_meta = doc_state.get("metadata")
            if isinstance(doc_meta, dict):
                metadata.update(doc_meta)
            stored_figures = doc_state.get("figures") or []
            for entry in stored_figures:
                if not isinstance(entry, dict):
                    continue
                try:
                    figure_state = FigureState(
                        index=int(entry["index"]),
                        page=entry.get("page"),
                        caption=str(entry.get("caption", "") or ""),
                        jpeg_b64=str(entry.get("jpeg_b64", "") or ""),
                        description=str(entry.get("description", "") or ""),
                        completed=bool(entry.get("completed", False)),
                    )
                except Exception:
                    continue
                try:
                    jpeg_bytes = figure_state.to_bytes()
                except Exception:
                    continue
                aggregated_figures.append(
                    Figure(
                        index=figure_state.index,
                        page=figure_state.page if isinstance(figure_state.page, int) else None,
                        caption=figure_state.caption,
                        jpeg_bytes=jpeg_bytes,
                    )
                )
                if figure_state.description:
                    aggregated_descs[figure_state.index] = FigureDesc(
                        index=figure_state.index,
                        page=figure_state.page if isinstance(figure_state.page, int) else None,
                        caption=figure_state.caption,
                        description=figure_state.description,
                    )
                    existing_fig_descs.pop(figure_state.index, None)
            stored_pages_total = doc_state.get("pages_total")
            if isinstance(stored_pages_total, int):
                total_pages = stored_pages_total
            stored_next_page = doc_state.get("next_page")
            if isinstance(stored_next_page, int):
                next_page = max(1, stored_next_page)
        if aggregated_base_md:
            log.info(
                "Doc parser: loaded Docling checkpoint from cache "
                f"(figures={len(aggregated_figures)} pages={total_pages})"
            )
        cache_tracker.resume()
        if isinstance(total_pages, int) and next_page > total_pages:
            doc_loaded_from_cache = True
    if isinstance(total_pages, int) and total_pages > 0:
        percent_batch = int(math.ceil(total_pages * max(PARSE_PAGE_BATCH_PERCENT, 0.0) / 100.0))
        candidates = [PARSE_PAGE_BATCH_SIZE]
        if percent_batch > 0:
            candidates.append(percent_batch)
        batch_size = max(1, min([total_pages] + candidates))
    else:
        batch_size = PARSE_PAGE_BATCH_SIZE
    total_batches = (
        math.ceil(total_pages / batch_size) if isinstance(total_pages, int) and total_pages > 0 else None
    )

    if cache_tracker:
        if isinstance(total_pages, int):
            if isinstance(next_page, int):
                pages_processed_seed = min(max(next_page - 1, 0), total_pages)
            else:
                pages_processed_seed = 0
        else:
            pages_processed_seed = None
        if total_batches and total_batches > 0 and isinstance(next_page, int) and isinstance(total_pages, int):
            batch_index_seed = min(total_batches, max(0, (max(next_page, 1) - 1) // batch_size))
        elif total_batches:
            batch_index_seed = 0
        else:
            batch_index_seed = None
        cache_tracker.update_phase(
            "docling",
            pages_processed=pages_processed_seed,
            pages_total=total_pages,
            figures_total=len(aggregated_figures),
            batch_index=batch_index_seed,
            batches_total=total_batches,
        )

    if not doc_loaded_from_cache:
        if isinstance(total_pages, int) and total_pages > batch_size:
            start_page = max(next_page, 1)
            figure_counter = aggregated_figures[-1].index + 1 if aggregated_figures else 1
            effective_total_batches = total_batches or max(1, math.ceil(total_pages / batch_size))
            completed_batches = max(0, (start_page - 1) // batch_size)
            batch_index = completed_batches + 1 if start_page <= total_pages else completed_batches
            if start_page <= 1 and not aggregated_base_md:
                log.info(
                    f"Doc parser: starting Docling conversion (batched) "
                    f"batch_size={batch_size} total_batches={effective_total_batches}"
                )
            while start_page <= total_pages:
                end_page = min(start_page + batch_size - 1, total_pages)
                t_doc = time.perf_counter()
                doc, chunk_md, chunk_figures, chunk_page_images = parser.parse(
                    tmp_path, page_range=(start_page, end_page)
                )
                doc_seconds = time.perf_counter() - t_doc
                overall_elapsed = time.perf_counter() - overall_start
                percent_complete = (end_page / total_pages) * 100 if total_pages else None
                batch_label = (
                    f"{batch_index}/{effective_total_batches}"
                    if effective_total_batches
                    else str(batch_index)
                )
                progress_text = (
                    f" progress={percent_complete:.1f}%"
                    if percent_complete is not None
                    else ""
                )
                log.info(
                    f"Doc parser: batch {batch_label} pages {start_page}-{end_page} "
                    f"took={doc_seconds:.2f}s elapsed_total={overall_elapsed:.2f}s "
                    f"(figures_detected={len(chunk_figures)}){progress_text}"
                )
                if not metadata.get("page_count"):
                    meta_extra = extract_metadata_from_doc(doc)
                    metadata.update(meta_extra)
                metadata["page_count"] = total_pages
                chunk_text = chunk_md.strip()
                if chunk_text:
                    aggregated_base_md = (
                        f"{aggregated_base_md}\n\n{chunk_text}"
                        if aggregated_base_md
                        else chunk_text
                    )
                chunk_length = end_page - start_page + 1
                is_first_batch = batch_index == 1
                has_text = bool(chunk_text)
                all_pages_have_images = len(chunk_page_images) >= chunk_length
                if (
                    is_first_batch
                    and not has_text
                    and all_pages_have_images
                    and not doc_loaded_from_cache
                ):
                    log.info(
                        "Doc parser: detected image-only document after first batch; switching to OCR pipeline"
                    )
                    image_only_doc = True
                    total_batches = effective_total_batches
                    aggregated_base_md = ""
                    aggregated_figures.clear()
                    aggregated_page_images = list(chunk_page_images)
                    if cache_tracker:
                        cache_tracker.update_phase(
                            "ocr",
                            pages_processed=end_page,
                            pages_total=total_pages,
                            figures_described=0,
                            figures_total=0,
                            batch_index=batch_index,
                            batches_total=effective_total_batches,
                        )
                    break
                batch_added_figs: List[Figure] = []
                new_valid_figs: List[Figure] = []
                for fig in chunk_figures:
                    page_no = fig.page
                    if page_no is not None and start_page > 1 and page_no <= chunk_length:
                        page_no += start_page - 1
                    figure = Figure(
                        index=figure_counter,
                        page=page_no,
                        caption=fig.caption,
                        jpeg_bytes=fig.jpeg_bytes,
                    )
                    aggregated_figures.append(figure)
                    batch_added_figs.append(figure)
                    if len(figure.jpeg_bytes) >= 5000:
                        if figure.index in aggregated_descs:
                            pass
                        elif figure.index in existing_fig_descs:
                            desc_text = existing_fig_descs.pop(figure.index)
                            desc_obj = FigureDesc(
                                index=figure.index,
                                page=figure.page,
                                caption=figure.caption,
                                description=desc_text,
                            )
                            aggregated_descs[figure.index] = desc_obj
                            if cache_tracker:
                                cache_tracker.record_description(figure.index, desc_text)
                        else:
                            new_valid_figs.append(figure)
                    figure_counter += 1
                if new_valid_figs:
                    try:
                        descs = _describe_figures(
                            new_valid_figs,
                            prompt,
                            job_key,
                            filename,
                            cache_tracker=cache_tracker,
                        )
                        for desc in descs:
                            aggregated_descs[desc.index] = desc
                            if cache_tracker:
                                cache_tracker.record_description(desc.index, desc.description or "")
                    except Exception as exc:
                        log.warning(f"Doc parser: figure description failed for batch {batch_index}: {exc}")
                for page_no, img_bytes in chunk_page_images:
                    if start_page > 1 and page_no <= chunk_length:
                        page_no += start_page - 1
                    aggregated_page_images.append((page_no, img_bytes))
                if cache_tracker:
                    cache_metadata = dict(metadata)
                    cache_metadata["page_count"] = total_pages
                    next_chunk_page = end_page + 1 if end_page < total_pages else total_pages + 1
                    figure_states: List[FigureState] = []
                    for fig in aggregated_figures:
                        state = FigureState.from_bytes(fig.index, fig.page, fig.caption, fig.jpeg_bytes)
                        desc_obj = aggregated_descs.get(fig.index)
                        if desc_obj and desc_obj.description:
                            state.description = desc_obj.description
                            state.completed = True
                        figure_states.append(state)
                    cache_tracker.store_doc_output(
                        base_markdown=aggregated_base_md,
                        metadata=cache_metadata,
                        figures=figure_states,
                        next_page=next_chunk_page,
                        pages_processed=end_page,
                        pages_total=total_pages,
                        batch_index=batch_index,
                        batches_total=effective_total_batches,
                    )
                start_page = end_page + 1
                batch_index += 1
            total_batches = effective_total_batches
        else:
            log.info("Doc parser: starting Docling conversion")
            t_doc = time.perf_counter()
            doc, chunk_md, chunk_figures, chunk_page_images = parser.parse(tmp_path)
            doc_seconds = time.perf_counter() - t_doc
            overall_elapsed = time.perf_counter() - overall_start
            log.info(
                f"Doc parser: conversion finished in {doc_seconds:.2f}s "
                f"(figures_detected={len(chunk_figures)}) elapsed_total={overall_elapsed:.2f}s"
            )
            meta_extra = extract_metadata_from_doc(doc)
            metadata.update(meta_extra)
            aggregated_base_md = chunk_md
            aggregated_figures = []
            aggregated_page_images = list(chunk_page_images)
            figure_counter = 1
            new_valid_figs: List[Figure] = []
            for fig in chunk_figures:
                figure = Figure(
                    index=figure_counter,
                    page=fig.page,
                    caption=fig.caption,
                    jpeg_bytes=fig.jpeg_bytes,
                )
                aggregated_figures.append(figure)
                if len(figure.jpeg_bytes) >= 5000:
                    if figure.index in aggregated_descs:
                        pass
                    elif figure.index in existing_fig_descs:
                        desc_text = existing_fig_descs.pop(figure.index)
                        desc_obj = FigureDesc(
                            index=figure.index,
                            page=figure.page,
                            caption=figure.caption,
                            description=desc_text,
                        )
                        aggregated_descs[figure.index] = desc_obj
                        if cache_tracker:
                            cache_tracker.record_description(figure.index, desc_text)
                    else:
                        new_valid_figs.append(figure)
                figure_counter += 1
            if new_valid_figs:
                try:
                    descs = _describe_figures(
                        new_valid_figs,
                        prompt,
                        job_key,
                        filename,
                        cache_tracker=cache_tracker,
                    )
                    for desc in descs:
                        aggregated_descs[desc.index] = desc
                        if cache_tracker:
                            cache_tracker.record_description(desc.index, desc.description or "")
                except Exception as exc:
                    log.warning(f"Doc parser: figure description failed: {exc}")
            total_pages = total_pages or metadata.get("page_count")
            total_batches = 1 if isinstance(total_pages, int) and total_pages and total_pages > 0 else 1
            if cache_tracker:
                cache_metadata = dict(metadata)
                cache_metadata["page_count"] = total_pages
                next_chunk_page = (
                    (total_pages + 1)
                    if isinstance(total_pages, int)
                    else len(aggregated_page_images) + 1
                )
                figure_states: List[FigureState] = []
                for fig in aggregated_figures:
                    state = FigureState.from_bytes(fig.index, fig.page, fig.caption, fig.jpeg_bytes)
                    desc_obj = aggregated_descs.get(fig.index)
                    if desc_obj and desc_obj.description:
                        state.description = desc_obj.description
                        state.completed = True
                    figure_states.append(state)
                cache_tracker.store_doc_output(
                    base_markdown=aggregated_base_md,
                    metadata=cache_metadata,
                    figures=figure_states,
                    next_page=next_chunk_page,
                    pages_processed=total_pages if isinstance(total_pages, int) else None,
                    pages_total=total_pages if isinstance(total_pages, int) else None,
                    batch_index=1,
                    batches_total=total_batches,
                )

    base_md = aggregated_base_md
    figures = aggregated_figures

    if total_pages is not None:
        metadata["page_count"] = total_pages

    try:
        file_size = os.path.getsize(tmp_path)
    except Exception:
        file_size = None
    if file_size is not None:
        metadata["file_size_bytes"] = file_size
    log.info(
        "Doc parser: file stats "
        f"size_bytes={file_size} pages={metadata.get('page_count')} expected_chunks=unknown"
    )
    if image_only_doc:
        log.info("Doc parser: running OCR-only pipeline")
        pages_for_ocr: List[Tuple[int, bytes]] = _render_pdf_pages_fallback(
            tmp_path,
            dpi=OCR_RENDER_DPI,
            max_pages=OCR_RENDER_MAX_PAGES,
            log=log,
        )
        if not pages_for_ocr:
            pages_for_ocr = list(aggregated_page_images)
        if not pages_for_ocr:
            log.warning("Doc parser: OCR-only path failed to obtain page images")
            markdown = ""
            metadata["page_images_available"] = 0
        else:
            metadata["page_images_available"] = len(pages_for_ocr)
            ocr_markdown, ocr_meta = _ocr_document_pages(pages_for_ocr, job_key, filename)
            markdown = clean_markdown(ocr_markdown)
            metadata.update(ocr_meta)
        page_count = metadata.get("page_count") or (
            len(pages_for_ocr) if pages_for_ocr else metadata.get("page_count")
        )
        if isinstance(page_count, int):
            metadata["page_count"] = page_count
        metadata["images_count"] = 0
        pages_processed_value = page_count if isinstance(page_count, int) else None
        char_count, line_count = _calc_text_density(markdown)
        metadata["text_char_count"] = char_count
        metadata["text_non_empty_lines"] = line_count
        if cache_tracker:
            cache_tracker.update_phase(
                "ocr",
                pages_processed=pages_processed_value,
                pages_total=pages_processed_value,
                figures_described=0,
                figures_total=0,
                batch_index=total_batches,
                batches_total=total_batches,
            )
            cache_tracker.finalize(markdown, metadata)
        log.info("Doc parser: composed final markdown via OCR-only pipeline")
        return markdown, metadata
    valid_figs_all = [f for f in aggregated_figures if len(f.jpeg_bytes) >= 5000]
    metadata["images_count"] = len(valid_figs_all)
    metadata["page_images_available"] = len(aggregated_page_images)
    log.info(
        f"Doc parser: retained {len(valid_figs_all)} valid figures (>=5KB) from {len(aggregated_figures)} total"
    )
    descs = sorted(aggregated_descs.values(), key=lambda d: d.index)
    markdown = compose_markdown(aggregated_base_md, descs)
    markdown = clean_markdown(markdown)
    if cache_tracker:
        pages_total_val = metadata.get("page_count")
        pages_total_int = pages_total_val if isinstance(pages_total_val, int) else None
        cache_tracker.update_phase(
            "figures",
            pages_processed=pages_total_int,
            pages_total=pages_total_int,
            figures_described=len([d for d in descs if d.description]),
            figures_total=len(valid_figs_all),
            batch_index=total_batches if isinstance(total_batches, int) else None,
            batches_total=total_batches,
        )
    page_count = metadata.get("page_count") or (len(aggregated_page_images) if aggregated_page_images else None)
    if page_count and metadata.get("page_count") is None:
        metadata["page_count"] = page_count
    char_count, line_count = _calc_text_density(markdown)
    metadata["text_char_count"] = char_count
    metadata["text_non_empty_lines"] = line_count
    if _should_trigger_ocr(char_count, line_count, page_count):
        chars_per_page = (char_count / page_count) if page_count else 0
        lines_per_page = (line_count / page_count) if page_count else 0
        log.info(
            "Doc parser: detected low text density "
            f"(chars_per_page={chars_per_page:.1f}, lines_per_page={lines_per_page:.1f}); running OCR fallback"
        )
        if cache_tracker:
            page_total_val = metadata.get("page_count")
            pages_total = page_total_val if isinstance(page_total_val, int) else None
            cache_tracker.update_phase(
                "ocr",
                pages_processed=pages_total,
                pages_total=pages_total,
                figures_described=len(descs),
                figures_total=len(valid_figs_all),
                batch_index=total_batches if isinstance(total_batches, int) else None,
                batches_total=total_batches,
            )
        pages_for_ocr: List[Tuple[int, bytes]] = list(aggregated_page_images)
        if not pages_for_ocr:
            fallback_images = _render_pdf_pages_fallback(
                tmp_path,
                dpi=OCR_RENDER_DPI,
                max_pages=OCR_RENDER_MAX_PAGES,
                log=log,
            )
            if fallback_images:
                pages_for_ocr = fallback_images
                metadata["page_images_available"] = len(fallback_images)
                log.info(f"Doc parser: generated {len(fallback_images)} fallback page renders for OCR")
            else:
                log.warning(
                    "Doc parser: low text density detected but no page images available for OCR fallback"
                )
        if pages_for_ocr:
            ocr_markdown, ocr_meta = _ocr_document_pages(pages_for_ocr, job_key, filename)
            if ocr_markdown:
                markdown = clean_markdown(ocr_markdown)
                metadata.update(ocr_meta)
                char_count, line_count = _calc_text_density(markdown)
                metadata["text_char_count"] = char_count
                metadata["text_non_empty_lines"] = line_count
    if cache_tracker:
        page_total_val = metadata.get("page_count")
        pages_total = page_total_val if isinstance(page_total_val, int) else None
        cache_tracker.update_phase(
            "compose",
            pages_processed=pages_total,
            pages_total=pages_total,
            figures_described=len(descs),
            figures_total=len(valid_figs_all),
            batch_index=total_batches if isinstance(total_batches, int) else None,
            batches_total=total_batches,
        )
    log.info("Doc parser: composed final markdown")
    log.info(
        f"Parsed PDF: pages={metadata.get('page_count')} images={metadata.get('images_count')}"
    )
    return markdown, metadata


def _parse_image(data: bytes, filename: str, ext: str, job_key: str) -> Tuple[str, Dict[str, Any]]:
    """Describe a single image and return markdown and metadata.

    Image inputs bypass Docling.  A ``GeminiVLM`` is invoked directly to
    produce a description, which is inserted into a simple markdown stub.
    """
    log = get_logger(job=job_key, file=filename, phase="parse")
    metadata = {"filename": filename, "file_type": ext.lstrip("."), "page_count": 1, "images_count": 1, "author": None}
    vlm = GeminiVLM(VLMConfig())
    prompt = default_prompt()
    log.info("Doc parser: describing single image with Gemini VLM")
    description = ""
    try:
        description = vlm.describe(data, prompt)
    except Exception as exc:
        log.warning(f"VLM describe failed for image: {exc}")
    if not description or len(description.strip()) < 10:
        description = "[No description]"
    markdown = f"![{filename}]({filename})\n\n{description.strip()}"
    return clean_markdown(markdown), metadata


def parse_document(
    tmp_path: str,
    filename: str,
    ext: str,
    *,
    image_bytes: Optional[bytes] = None,
    job_key: str,
    cache_tracker: Optional[ParseProgressTracker] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Parse a document (PDF or image) and return markdown and metadata.

    :param tmp_path: Path on disk for PDFs.  Ignored for image inputs.
    :param filename: Original file name supplied by the client.
    :param ext: File extension (lowercase, including dot).
    :param image_bytes: Raw bytes of the image when ``ext`` is not `.pdf`.
    :param job_key: Unique identifier for the job, used for logging.
    :return: A tuple ``(markdown, metadata)``.
    """
    if ext == ".pdf":
        return _parse_pdf(tmp_path, filename, job_key, cache_tracker=cache_tracker)
    else:
        if image_bytes is None:
            # In synchronous code we assume the caller has already read the file
            with open(tmp_path, "rb") as f:
                data = f.read()
        else:
            data = image_bytes
        return _parse_image(data, filename, ext, job_key)


__all__ = [
    "parse_document",
    "extract_metadata_from_doc",
    "clean_markdown",
]







