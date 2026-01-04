"""Pydantic models for FastMCP server responses."""

from datetime import datetime

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    """A single search result."""

    title: str
    snippet: str
    url: str
    source: str
    source_url: str
    score: float


class DocumentResult(BaseModel):
    """Full document content with pagination support."""

    title: str
    content: str
    url: str
    source: str
    source_url: str
    offset: int = 0
    length: int = 0
    total_length: int = 0
    has_more: bool = False


class ExcerptItem(BaseModel):
    """A single excerpt from a document."""

    content: str
    start_pos: int
    end_pos: int
    score: float


class DocumentExcerptResult(BaseModel):
    """Document with relevant excerpts."""

    title: str
    url: str
    source: str
    source_url: str
    total_length: int
    excerpts: list[ExcerptItem]


class SourceInfo(BaseModel):
    """Information about a documentation source."""

    name: str
    url: str
    doc_count: int
    last_updated: datetime | None


class SourceRefreshStats(BaseModel):
    """Statistics for a single source after refresh."""

    name: str
    url: str
    doc_count: int
    errors: int


class RefreshResult(BaseModel):
    """Result of a refresh operation."""

    refreshed_count: int
    indexed_documents: int
    indexed_chunks: int
    sources: list[SourceRefreshStats]
    errors: list[str] | None = None
    skipped: bool = False
    reason: str | None = None
