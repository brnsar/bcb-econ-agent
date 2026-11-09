"""Build the local knowledge base: Copom document index + monthly indicator table."""

from __future__ import annotations

import logging
from datetime import date

from . import bcb_client
from .config import Settings
from .embeddings import Embedder
from .indicators import SERIES, IndicatorStore, build_monthly_rows
from .text import chunk_text, html_to_text
from .vector_store import VectorStore

log = logging.getLogger(__name__)

EMBED_BATCH = 16


def ingest_documents(settings: Settings, embedder: Embedder) -> VectorStore:
    statements = [m for m in bcb_client.list_statement_meetings() if m >= settings.first_meeting]
    minutes = [m for m in bcb_client.list_minutes_meetings() if m >= settings.first_meeting]
    log.info("Fetching %d statements and %d minutes", len(statements), len(minutes))

    docs = [bcb_client.get_statement(m) for m in sorted(statements)]
    docs += [bcb_client.get_minutes(m) for m in sorted(minutes)]

    records: list[dict] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(html_to_text(doc.html))):
            records.append(
                {
                    "id": f"{doc.doc_type}-{doc.meeting}-{i}",
                    "meeting": doc.meeting,
                    "doc_type": doc.doc_type,
                    "date": doc.date,
                    "title": doc.title,
                    "url": doc.url,
                    "text": chunk,
                }
            )

    store = VectorStore(embedder.name, embedder.dim)
    for start in range(0, len(records), EMBED_BATCH):
        batch = records[start : start + EMBED_BATCH]
        # Prefix the title so each chunk carries which meeting and document it came from.
        texts = [f"{r['title']} ({r['doc_type']}, {r['date']})\n{r['text']}" for r in batch]
        store.add(embedder.embed(texts), batch)
        log.info("Embedded %d/%d chunks", min(start + EMBED_BATCH, len(records)), len(records))

    store.save(settings.index_dir)
    return store


def ingest_indicators(settings: Settings, today: date | None = None) -> IndicatorStore:
    today = today or date.today()
    year, month = (int(x) for x in settings.indicators_start.split("-"))
    start = date(year, month, 1)
    series = {name: bcb_client.get_sgs_series(code, start, today) for name, code in SERIES.items()}
    rows = build_monthly_rows(series["usd"], series["selic"], series["ipca"])
    store = IndicatorStore(settings.indicators_db)
    store.write(rows)
    log.info("Wrote %d monthly rows to %s", len(rows), settings.indicators_db)
    return store
