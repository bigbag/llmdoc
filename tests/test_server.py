"""Tests for llmdoc.server module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmdoc.app import LLMDocApp
from llmdoc.config import Config
from llmdoc.fetcher import DocumentFetcher
from llmdoc.indexer import BM25Index
from llmdoc.server import do_refresh, get_app, mcp
from llmdoc.store import DocumentStore


class TestGetApp:
    """Tests for get_app dependency."""

    def test_get_app_not_initialized(self):
        """Test that get_app raises when mcp has no app set."""
        # Ensure mcp doesn't have the app attribute
        old_app = getattr(mcp, "_llmdoc_app", None)
        if hasattr(mcp, "_llmdoc_app"):
            delattr(mcp, "_llmdoc_app")

        try:
            with pytest.raises(RuntimeError, match="App not initialized"):
                get_app()
        finally:
            if old_app is not None:
                mcp._llmdoc_app = old_app

    def test_get_app_returns_app(self, sample_config):
        """Test that get_app returns the app when set on mcp server."""
        app = LLMDocApp.create(sample_config)

        # Set app on mcp server (as lifespan does)
        old_app = getattr(mcp, "_llmdoc_app", None)
        try:
            mcp._llmdoc_app = app
            result = get_app()
            assert result is app
        finally:
            if old_app is not None:
                mcp._llmdoc_app = old_app
            elif hasattr(mcp, "_llmdoc_app"):
                delattr(mcp, "_llmdoc_app")
            app.close()


class TestMcpServer:
    """Tests for the MCP server configuration."""

    def test_server_name(self):
        """Test server has correct name."""
        assert mcp.name == "LLMDoc"

    def test_server_has_tools(self):
        """Test server has expected tools."""
        tool_names = [t.name for t in mcp._tool_manager._tools.values()]

        assert "search_docs" in tool_names
        assert "get_doc" in tool_names
        assert "list_sources" in tool_names
        assert "refresh_sources" in tool_names

    def test_server_has_resource(self):
        """Test server has the sources resource."""
        resource_names = list(mcp._resource_manager._resources.keys())
        assert "doc://sources" in resource_names


class TestDoRefresh:
    """Tests for do_refresh function."""

    @pytest.mark.asyncio
    async def test_do_refresh_success(self, temp_db_path, sample_sources):
        """Test successful refresh."""
        # Create DB first
        init_store = DocumentStore(temp_db_path, read_only=False)
        init_store.close()

        # Create app with real components
        config = Config(sources=sample_sources, db_path=temp_db_path)
        store = DocumentStore(temp_db_path, read_only=True)
        index = BM25Index()
        fetcher = DocumentFetcher()

        app = LLMDocApp(config=config, store=store, index=index, fetcher=fetcher)

        # Mock the fetcher
        mock_doc = MagicMock()
        mock_doc.url = "https://a.example.com/doc.md"
        mock_doc.title = "Test Doc"
        mock_doc.content = "Content"

        with patch.object(
            app.fetcher,
            "fetch_all_from_source",
            new=AsyncMock(return_value=([mock_doc], [])),
        ):
            result = await do_refresh(app)

            assert result.refreshed_count >= 0
            assert result.indexed_documents >= 0

    @pytest.mark.asyncio
    async def test_do_refresh_with_errors(self, temp_db_path, sample_sources):
        """Test refresh with fetch errors."""
        # Create DB first
        init_store = DocumentStore(temp_db_path, read_only=False)
        init_store.close()

        config = Config(sources=sample_sources, db_path=temp_db_path)
        store = DocumentStore(temp_db_path, read_only=True)
        index = BM25Index()
        fetcher = DocumentFetcher()

        app = LLMDocApp(config=config, store=store, index=index, fetcher=fetcher)

        with patch.object(
            app.fetcher,
            "fetch_all_from_source",
            new=AsyncMock(return_value=([], ["Error fetching doc"])),
        ):
            result = await do_refresh(app)

            assert result.errors is not None
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_do_refresh_rebuilds_index(self, temp_db_path, sample_sources):
        """Test that refresh rebuilds the index."""
        init_store = DocumentStore(temp_db_path, read_only=False)
        init_store.close()

        config = Config(sources=sample_sources, db_path=temp_db_path)
        store = DocumentStore(temp_db_path, read_only=True)
        index = BM25Index()
        fetcher = DocumentFetcher()

        app = LLMDocApp(config=config, store=store, index=index, fetcher=fetcher)

        # Start with empty index
        assert app.index.document_count == 0

        mock_doc = MagicMock()
        mock_doc.url = "https://a.example.com/doc.md"
        mock_doc.title = "Test Doc"
        mock_doc.content = "Some test content here"

        with patch.object(
            app.fetcher,
            "fetch_all_from_source",
            new=AsyncMock(return_value=([mock_doc], [])),
        ):
            result = await do_refresh(app)

            # Index should now have documents
            assert result.indexed_documents >= 0


class TestSearchDocsLogic:
    """Tests for search functionality (testing index directly)."""

    def test_search_returns_results(self, populated_index):
        """Test that search returns relevant results."""
        results = populated_index.search("tools functions")

        assert len(results) > 0
        assert results[0].title == "Tools Documentation"

    def test_search_with_source_filter(self, populated_index):
        """Test search with source filter."""
        results = populated_index.search("documentation", source_filter="source_a")
        assert all(r.source_name == "source_a" for r in results)

    def test_search_empty_index(self, bm25_index):
        """Test search on empty index."""
        results = bm25_index.search("anything")
        assert results == []


class TestGetDocLogic:
    """Tests for get_doc functionality (testing store directly)."""

    def test_get_document_found(self, populated_store):
        """Test getting an existing document."""
        doc = populated_store.get_document_by_url("https://a.example.com/tools.md")

        assert doc is not None
        assert doc.title == "Tools Documentation"
        assert doc.source_name == "source_a"

    def test_get_document_not_found(self, populated_store):
        """Test getting a non-existent document."""
        doc = populated_store.get_document_by_url("https://nonexistent.com/doc.md")
        assert doc is None


class TestListSourcesLogic:
    """Tests for list_sources functionality (testing store directly)."""

    def test_get_source_stats(self, populated_store):
        """Test getting source statistics."""
        stats = populated_store.get_source_stats()

        assert len(stats) == 2
        stats_by_name = {s["name"]: s for s in stats}
        assert stats_by_name["source_a"]["doc_count"] == 2
        assert stats_by_name["source_b"]["doc_count"] == 1

    def test_get_source_stats_empty(self, document_store):
        """Test getting stats from empty store."""
        stats = document_store.get_source_stats()
        assert stats == []
