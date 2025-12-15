"""Command‑line entry point for the synchronous ingestion pipeline.

Running ``python -m llm_document_ingestion.src.main`` will invoke the
ingestion pipeline on the specified PDF or image and write an
enriched Markdown file alongside the input document.  The output
filename is derived from the input by appending ``_enriched.md``.

Usage:

.. code-block:: bash

    python -m llm_document_ingestion.src.main input.pdf --category finance --collection documents

See the README for details on environment variables and the pipeline.
"""

from __future__ import annotations

import os
import click
from dotenv import load_dotenv

from .pipeline import process_file


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--category", default="general", help="Category for the document")
@click.option("--collection", default=None, help="Qdrant collection name (overrides env)")
def cli(input_path: str, category: str, collection: str | None) -> None:
    """Process a document and write the enriched markdown to disk."""
    load_dotenv()
    filename = os.path.basename(input_path)
    ext = os.path.splitext(filename)[1].lower()
    collection_name = collection or os.getenv("QDRANT_COLLECTION", "documents")
    job_key = os.path.splitext(filename)[0] + "-cli"
    result = process_file(
        file_path=input_path,
        filename=filename,
        ext=ext,
        category=category,
        collection=collection_name,
        job_key=job_key,
    )
    out_path = os.path.splitext(input_path)[0] + "_enriched.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    click.echo(f"Saved enriched markdown to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    cli()
