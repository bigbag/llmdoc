"""FastMCP server for LLMDoc."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext, Depends
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from pydantic import Field

from .app import LLMDocApp
from .config import load_config
from .models import (
    DocumentExcerptResult,
    DocumentResult,
    ExcerptItem,
    RefreshResult,
    SearchResultItem,
    SourceInfo,
)
from .refresh import do_refresh, periodic_refresh, refresh_lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_app() -> LLMDocApp:
    """Get app from server (set during lifespan)."""
    app: LLMDocApp | None = getattr(mcp, "_llmdoc_app", None)
    if app is None:
        raise RuntimeError("App not initialized - server not started")
    return app


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Lifespan context manager for server startup/shutdown."""
    # Load configuration
    config = load_config()
    source_names = [s.name for s in config.sources]
    logger.info(f"Loaded configuration with {len(config.sources)} sources: {source_names}")

    if not config.sources:
        logger.warning(
            "No documentation sources configured. Set LLMDOC_SOURCES environment variable or create llmdoc.json"
        )

    # Initialize application
    app = LLMDocApp.create(config)
    logger.info(f"Initialized with {app.index.document_count} documents from {app.config.db_path}")

    # Store app on server for middleware access
    setattr(server, "_llmdoc_app", app)

    # Check if any source needs refresh based on TTL
    if config.skip_startup_refresh:
        logger.info("Startup refresh disabled via config (skip_startup_refresh=True)")
    elif config.sources:
        stats = app.store.get_source_stats()
        stats_by_name = {s["name"]: s for s in stats}
        threshold = datetime.now() - timedelta(hours=config.refresh_interval_hours)

        logger.info(
            f"Staleness check: {len(stats)} sources in DB, "
            f"threshold={threshold}, refresh_interval={config.refresh_interval_hours}h"
        )
        for s in stats:
            logger.info(f"  DB source: name='{s['name']}', last_updated={s['last_updated']}")

        # Determine if refresh is needed:
        # - If we have NO data at all, refresh to populate
        # - If any source WITH data is stale, refresh
        # - Sources that never fetched successfully don't force refresh of others
        needs_refresh = False
        has_any_data = len(stats) > 0

        if not has_any_data:
            logger.info("No documents in database, triggering initial fetch")
            needs_refresh = True
        else:
            for source in config.sources:
                source_stat = stats_by_name.get(source.name)
                if not source_stat or not source_stat.get("last_updated"):
                    # Source never fetched - will be attempted during refresh
                    # but don't force refresh just for this
                    logger.info(f"Source '{source.name}' has no data in DB (may have failed to fetch)")
                elif source_stat["last_updated"] < threshold:
                    logger.info(
                        f"Source '{source.name}' is stale "
                        f"(last updated: {source_stat['last_updated']}, threshold: {threshold})"
                    )
                    needs_refresh = True
                    break
                else:
                    logger.info(f"Source '{source.name}' is fresh (last updated: {source_stat['last_updated']})")

        if needs_refresh:
            logger.info("Triggering startup refresh...")
            try:
                await do_refresh(app)
            except Exception as e:
                logger.error(f"Startup refresh failed: {e}")
        else:
            logger.info("All sources with data are fresh, skipping startup refresh")

    # Start periodic refresh task
    refresh_task = asyncio.create_task(periodic_refresh(app))

    try:
        yield
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

        app.close()
        delattr(server, "_llmdoc_app")
        logger.info("Server shutdown complete")


mcp = FastMCP(
    name="LLMDoc",
    instructions="""
    LLMDoc provides documentation search across configured llms.txt sources.

    Use the search_docs tool to find relevant documentation based on your query.
    You can optionally filter by source name (e.g., 'fast_mcp', 'pydantic_ai').
    Results include source names and URLs for attribution.

    Use get_doc to retrieve the full content of a document by its URL.
    Use get_doc_excerpt for large documents when you only need specific sections -
    it returns targeted excerpts based on a query instead of the full content.
    Use list_sources to see what documentation sources are available.
    Use refresh_sources to manually trigger a refresh of all documentation.
    """,
    lifespan=lifespan,
)


@mcp.tool(
    annotations={
        "title": "Search Documentation",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"search", "documentation"},
)
async def search_docs(
    query: Annotated[str, Field(description="The search query to find relevant documentation")],
    limit: Annotated[int, Field(description="Maximum number of results to return", ge=1, le=50)] = 5,
    source: Annotated[
        str | None,
        Field(description="Optional source name to filter results (e.g., 'fast_mcp', 'pydantic_ai')"),
    ] = None,
    ctx: Context = CurrentContext(),
    app: LLMDocApp = Depends(get_app),
) -> list[SearchResultItem]:
    """Search documentation and return relevant passages with source URLs.

    Use this tool when you need to find information about specific topics,
    APIs, or concepts. The search uses BM25 ranking for relevance.

    Args:
        query: The search query to find relevant documentation.
        limit: Maximum number of results to return (default: 5).
        source: Optional source name to filter results (e.g., 'fast_mcp', 'pydantic_ai').

    Returns:
        List of search results with title, snippet, url, source (name), source_url, and score.
    """
    await ctx.debug(f"Searching for: {query}")
    results = app.index.search(query, limit=limit, source_filter=source)

    return [
        SearchResultItem(
            title=r.title or "Untitled",
            snippet=r.snippet,
            url=r.doc_url,
            source=r.source_name,
            source_url=r.source_url,
            score=round(r.score, 4),
        )
        for r in results
    ]


DEFAULT_CHUNK_SIZE = 50_000  # 50KB


@mcp.tool(
    annotations={
        "title": "Get Document",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"retrieval", "documentation"},
)
async def get_doc(
    url: Annotated[str, Field(description="The URL of the document (as returned by search_docs)")],
    offset: Annotated[int, Field(description="Start position in bytes (for pagination)", ge=0)] = 0,
    limit: Annotated[
        int,
        Field(
            description="Max bytes to return (default 50000, max 100000)",
            ge=1000,
            le=100_000,
        ),
    ] = DEFAULT_CHUNK_SIZE,
    ctx: Context = CurrentContext(),
    app: LLMDocApp = Depends(get_app),
) -> DocumentResult:
    """Get document content with pagination support for large documents.

    For documents larger than 50KB, use offset/limit to paginate through content.
    The response includes has_more=True if more content is available.
    For targeted retrieval, use get_doc_excerpt instead.

    Args:
        url: The URL of the document (as returned by search_docs).
        offset: Start position in bytes (default: 0).
        limit: Max bytes to return per call (default: 50000, max: 100000).

    Returns:
        Document with content chunk, pagination metadata (offset, length, total_length, has_more).
    """
    async with refresh_lock:
        doc = app.store.get_document_by_url(url)

    if not doc:
        raise ToolError(f"Document not found: {url}")

    total_length = len(doc.content)
    chunk = doc.content[offset : offset + limit]

    return DocumentResult(
        title=doc.title or "Untitled",
        content=chunk,
        url=doc.doc_url,
        source=doc.source_name,
        source_url=doc.source_url,
        offset=offset,
        length=len(chunk),
        total_length=total_length,
        has_more=(offset + len(chunk)) < total_length,
    )


@mcp.tool(
    annotations={
        "title": "Get Document Excerpt",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"retrieval", "documentation"},
)
async def get_doc_excerpt(
    url: Annotated[str, Field(description="The URL of the document")],
    query: Annotated[str, Field(description="Query to find relevant sections within the document")],
    max_chunks: Annotated[int, Field(description="Maximum chunks to return", ge=1, le=20)] = 5,
    context_chars: Annotated[int, Field(description="Extra context chars around each chunk", ge=0, le=2000)] = 500,
    ctx: Context = CurrentContext(),
    app: LLMDocApp = Depends(get_app),
) -> DocumentExcerptResult:
    """Get relevant excerpts from a large document matching a query.

    Use this instead of get_doc for large documents. Returns targeted excerpts
    based on BM25 relevance to your query.

    Args:
        url: The URL of the document.
        query: Query to find relevant sections within the document.
        max_chunks: Maximum number of chunks to return (default: 5).
        context_chars: Extra context characters around each chunk (default: 500).

    Returns:
        Document metadata with list of relevant excerpts, each containing
        content, position, and relevance score.
    """
    # Acquire lock briefly to ensure store connection is valid
    async with refresh_lock:
        doc = app.store.get_document_by_url(url)
    if not doc:
        raise ToolError(f"Document not found: {url}")

    # Search within document
    results = app.index.search_within_document(url, query, top_k=max_chunks)
    if not results:
        raise ToolError(f"No relevant excerpts found for query: {query}")

    # Expand context and build excerpts
    excerpts: list[ExcerptItem] = []
    content = doc.content
    for chunk, score in results:
        start = max(0, chunk.start_pos - context_chars)
        end = min(len(content), chunk.start_pos + len(chunk.content) + context_chars)

        excerpts.append(
            ExcerptItem(
                content=content[start:end],
                start_pos=start,
                end_pos=end,
                score=round(score, 4),
            )
        )

    return DocumentExcerptResult(
        title=doc.title or "Untitled",
        url=doc.doc_url,
        source=doc.source_name,
        source_url=doc.source_url,
        total_length=len(content),
        excerpts=excerpts,
    )


@mcp.tool(
    annotations={
        "title": "List Sources",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"metadata", "sources"},
)
async def list_sources(
    ctx: Context = CurrentContext(),
    app: LLMDocApp = Depends(get_app),
) -> list[SourceInfo]:
    """List all configured documentation sources with their statistics.

    Use this to discover what documentation sources are available for searching.
    Each source has a name that can be used to filter search_docs results.

    Returns:
        List of sources with name, url, doc_count, and last_updated.
    """
    # Acquire lock briefly to ensure store connection is valid
    async with refresh_lock:
        stats = app.store.get_source_stats()
    stats_by_name = {s["name"]: s for s in stats}

    return [
        SourceInfo(
            name=source.name,
            url=source.url,
            doc_count=stats_by_name.get(source.name, {}).get("doc_count", 0),
            last_updated=stats_by_name.get(source.name, {}).get("last_updated"),
        )
        for source in app.config.sources
    ]


@mcp.tool(
    annotations={
        "title": "Refresh Sources",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    tags={"admin", "refresh"},
)
async def refresh_sources(
    ctx: Context = CurrentContext(),
    app: LLMDocApp = Depends(get_app),
) -> RefreshResult:
    """Manually trigger a refresh of all documentation sources.

    This fetches documentation from all configured llms.txt URLs and updates
    the local index. Use this when you need the latest documentation content.

    Returns:
        Dictionary with refreshed_count, indexed_documents, indexed_chunks, sources, and any errors.
    """
    logger.info("Starting manual refresh...")
    return await do_refresh(app)


@mcp.resource(
    "doc://sources",
    annotations={"audience": ["user", "assistant"]},
)
def get_sources_resource(
    app: LLMDocApp = Depends(get_app),
) -> dict[str, Any]:
    """Provides list of documentation sources as a resource.

    Returns configured sources and refresh interval settings.
    """
    return {
        "sources": [{"name": s.name, "url": s.url} for s in app.config.sources],
        "refresh_interval_hours": app.config.refresh_interval_hours,
    }


def main() -> None:
    """Main entry point for the server."""
    mcp.run()


if __name__ == "__main__":
    main()
