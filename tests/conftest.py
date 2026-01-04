"""Shared test fixtures for LLMDoc tests."""

from datetime import datetime
from pathlib import Path

import pytest

from llmdoc.config import Config, Source
from llmdoc.fetcher import DocumentFetcher
from llmdoc.indexer import BM25Index
from llmdoc.store import Document, DocumentStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provide a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def sample_source() -> Source:
    """Provide a sample source."""
    return Source(name="test_source", url="https://example.com/llms.txt")


@pytest.fixture
def sample_sources() -> list[Source]:
    """Provide sample sources."""
    return [
        Source(name="source_a", url="https://a.example.com/llms.txt"),
        Source(name="source_b", url="https://b.example.com/llms.txt"),
    ]


@pytest.fixture
def sample_config(temp_db_path: str, sample_sources: list[Source]) -> Config:
    """Provide a sample configuration."""
    return Config(
        sources=sample_sources,
        db_path=temp_db_path,
        refresh_interval_hours=1,
        max_concurrent_fetches=2,
    )


@pytest.fixture
def sample_document() -> Document:
    """Provide a sample document."""
    return Document(
        id=1,
        source_name="test_source",
        source_url="https://example.com/llms.txt",
        doc_url="https://example.com/doc1.md",
        title="Test Document",
        content="This is a test document with some content.\n\nIt has multiple paragraphs.",
        content_hash="abc123",
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide multiple sample documents."""
    now = datetime.now()
    return [
        Document(
            id=1,
            source_name="source_a",
            source_url="https://a.example.com/llms.txt",
            doc_url="https://a.example.com/tools.md",
            title="Tools Documentation",
            content="# Tools\n\nThis document covers tools and how to use them.\n\nTools are functions that can be called by the LLM.",
            content_hash="hash1",
            updated_at=now,
        ),
        Document(
            id=2,
            source_name="source_a",
            source_url="https://a.example.com/llms.txt",
            doc_url="https://a.example.com/resources.md",
            title="Resources Documentation",
            content="# Resources\n\nResources expose data to the LLM.\n\nThey can be static or dynamic.",
            content_hash="hash2",
            updated_at=now,
        ),
        Document(
            id=3,
            source_name="source_b",
            source_url="https://b.example.com/llms.txt",
            doc_url="https://b.example.com/agents.md",
            title="Agents Documentation",
            content="# Agents\n\nAgents are the primary interface for interacting with LLMs.\n\nThey manage context and tool calls.",
            content_hash="hash3",
            updated_at=now,
        ),
    ]


@pytest.fixture
def document_store(temp_db_path: str) -> DocumentStore:
    """Provide a temporary document store."""
    store = DocumentStore(temp_db_path, read_only=False)
    yield store
    store.close()


@pytest.fixture
def populated_store(document_store: DocumentStore, sample_documents: list[Document]) -> DocumentStore:
    """Provide a document store with sample data."""
    for doc in sample_documents:
        document_store.upsert_document(
            source_name=doc.source_name,
            source_url=doc.source_url,
            doc_url=doc.doc_url,
            title=doc.title,
            content=doc.content,
        )
    return document_store


@pytest.fixture
def bm25_index() -> BM25Index:
    """Provide a BM25 index."""
    return BM25Index(chunk_size=200, chunk_overlap=50)


@pytest.fixture
def populated_index(bm25_index: BM25Index, sample_documents: list[Document]) -> BM25Index:
    """Provide a BM25 index with sample data."""
    bm25_index.build_index(sample_documents)
    return bm25_index


@pytest.fixture
def document_fetcher() -> DocumentFetcher:
    """Provide a document fetcher."""
    return DocumentFetcher(timeout=10.0, max_concurrent=2)


@pytest.fixture
def sample_llms_txt() -> str:
    """Provide sample llms.txt content."""
    return """# Example Project

> A sample project for testing.

## Documentation

- [Getting Started](getting-started.md): How to get started
- [API Reference](api/reference.md): Full API documentation
- [Examples](examples/index.md)
"""


@pytest.fixture
def sample_markdown() -> str:
    """Provide sample markdown content."""
    return """# Getting Started

Welcome to the getting started guide.

## Installation

Install using pip:

```bash
pip install example
```

## Usage

Here's how to use it:

```python
import example
example.run()
```
"""
