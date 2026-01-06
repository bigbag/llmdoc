"""Refresh logic for LLMDoc documentation sources."""

import asyncio
import contextlib
import fcntl
import logging
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .app import LLMDocApp
from .models import RefreshResult, SourceRefreshStats
from .store import DocumentStore

logger = logging.getLogger(__name__)

# Lock to prevent read operations during the brief swap phase
refresh_lock = asyncio.Lock()


@contextmanager
def file_lock(lock_path: str) -> Generator[bool, None, None]:
    """Cross-process file lock using fcntl.

    Acquires an exclusive lock on the lock file. If another process holds
    the lock, yields False immediately (non-blocking).

    Args:
        lock_path: Path to the lock file.

    Yields:
        True if lock was acquired, False if another process holds it.
    """
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    lock_file = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield True  # Lock acquired
    except BlockingIOError:
        yield False  # Another process has the lock
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


async def _fetch_all_sources(
    app: LLMDocApp,
) -> tuple[list[tuple[Any, Any, list[str]]], list[str]]:
    """Fetch documents from all configured sources.

    This is the fetch phase - runs without holding the refresh lock since
    it's async and can take time.

    Args:
        app: The application instance.

    Returns:
        Tuple of (fetched_data, errors) where fetched_data is a list of
        (source, documents, source_errors) tuples.
    """
    fetched_data: list[tuple[Any, Any, list[str]]] = []
    all_errors: list[str] = []

    for source in app.config.sources:
        logger.info(f"Fetching source: {source.name} ({source.url})")
        try:
            documents, source_errors = await app.fetcher.fetch_all_from_source(source.url)
            fetched_data.append((source, documents, source_errors))
            all_errors.extend(source_errors)
        except Exception as e:
            error_msg = f"Failed to fetch {source.name}: {e}"
            logger.error(error_msg)
            all_errors.append(error_msg)

    return fetched_data, all_errors


def _write_source_to_store(
    write_store: DocumentStore,
    source: Any,
    documents: list,
    source_errors: list[str],
) -> tuple[int, SourceRefreshStats, list[str]]:
    """Write documents for a single source to the store.

    Args:
        write_store: The write-enabled document store.
        source: The source configuration.
        documents: List of fetched documents.
        source_errors: Errors from fetching this source.

    Returns:
        Tuple of (doc_count, stats, errors).
    """
    errors: list[str] = []
    doc_count = 0
    valid_urls: set[str] = set()

    try:
        for doc in documents:
            write_store.upsert_document(
                source_name=source.name,
                source_url=source.url,
                doc_url=doc.url,
                title=doc.title,
                content=doc.content,
            )
            valid_urls.add(doc.url)
            doc_count += 1

        # Remove stale documents
        deleted = write_store.delete_stale_documents(source.name, valid_urls)
        if deleted:
            logger.info(f"Removed {deleted} stale documents from {source.name}")

    except Exception as e:
        error_msg = f"Failed to update store for {source.name}: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    stats = SourceRefreshStats(
        name=source.name,
        url=source.url,
        doc_count=len(documents),
        errors=len([e for e in source_errors if source.url in e]),
    )

    return doc_count, stats, errors


def _rebuild_index(app: LLMDocApp) -> None:
    """Rebuild the BM25 index from current store data.

    Args:
        app: The application instance.
    """
    all_docs = app.store.get_all_documents()
    app.index.build_index(all_docs)
    logger.info(f"Index rebuilt with {app.index.document_count} documents, {app.index.chunk_count} chunks")


async def do_refresh(app: LLMDocApp) -> RefreshResult:
    """Refresh all configured documentation sources.

    Uses a shadow database pattern for zero-downtime updates:
    1. Fetch all sources (no lock)
    2. Write to temporary database (file lock for cross-process safety)
    3. Atomic swap of temp DB to primary (brief asyncio lock)
    4. Rebuild in-memory index

    Read operations are only blocked during the brief swap phase (~10ms),
    not during the entire fetch/write operation.

    Args:
        app: The application instance.

    Returns:
        RefreshResult with refresh statistics.
    """
    lock_path = app.config.db_path + ".lock"
    temp_db_path = app.config.db_path + ".tmp"

    # Phase 1: Fetch all sources (no lock needed)
    fetched_data, all_errors = await _fetch_all_sources(app)

    # Phase 2-3: Write to temp DB + atomic swap (with file lock)
    source_stats: list[SourceRefreshStats] = []
    total_docs = 0

    with file_lock(lock_path) as acquired:
        if not acquired:
            logger.info("Refresh locked by another instance, skipping")
            return RefreshResult(
                refreshed_count=0,
                indexed_documents=app.index.document_count,
                indexed_chunks=app.index.chunk_count,
                sources=[],
                skipped=True,
                reason="Refresh locked by another instance",
            )

        # Copy current DB as base for temp (if exists)
        if os.path.exists(app.config.db_path):
            shutil.copy2(app.config.db_path, temp_db_path)

        try:
            # Write to temp database
            write_store = DocumentStore(temp_db_path, read_only=False)

            for source, documents, source_errors in fetched_data:
                doc_count, stats, errors = _write_source_to_store(write_store, source, documents, source_errors)
                total_docs += doc_count
                source_stats.append(stats)
                all_errors.extend(errors)

            write_store.close()

        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
            raise

        # Phase 3: Atomic swap (brief asyncio lock for in-process read safety)
        async with refresh_lock:
            app.store.close()
            os.replace(temp_db_path, app.config.db_path)
            app.store = DocumentStore(app.config.db_path, read_only=True)

    # Phase 4: Rebuild index (outside locks)
    _rebuild_index(app)

    return RefreshResult(
        refreshed_count=total_docs,
        indexed_documents=app.index.document_count,
        indexed_chunks=app.index.chunk_count,
        sources=source_stats,
        errors=all_errors if all_errors else None,
    )


async def periodic_refresh(app: LLMDocApp) -> None:
    """Background task for periodic refresh.

    Args:
        app: The application instance.
    """
    interval_seconds = app.config.refresh_interval_hours * 3600
    logger.info(f"Periodic refresh enabled: every {app.config.refresh_interval_hours} hours")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.info("Starting scheduled refresh...")
            result = await do_refresh(app)
            if result.skipped:
                logger.info(f"Refresh skipped: {result.reason}")
            else:
                logger.info(f"Scheduled refresh completed: {result.refreshed_count} docs")
        except asyncio.CancelledError:
            logger.info("Periodic refresh cancelled")
            raise
        except Exception as e:
            logger.error(f"Scheduled refresh failed: {e}")
