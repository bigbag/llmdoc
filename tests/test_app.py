"""Tests for llmdoc.app module."""

from pathlib import Path

from llmdoc.app import LLMDocApp
from llmdoc.config import Config, Source


class TestLLMDocApp:
    """Tests for LLMDocApp class."""

    def test_create_app(self, sample_config):
        """Test creating an app instance."""
        app = LLMDocApp.create(sample_config)

        try:
            assert app.config == sample_config
            assert app.store is not None
            assert app.index is not None
            assert app.fetcher is not None
        finally:
            app.close()

    def test_create_app_creates_db(self, tmp_path):
        """Test that create() creates the database if it doesn't exist."""
        db_path = str(tmp_path / "new_db" / "test.db")
        config = Config(
            sources=[Source(name="test", url="https://example.com/llms.txt")],
            db_path=db_path,
        )

        app = LLMDocApp.create(config)
        try:
            assert Path(db_path).exists()
        finally:
            app.close()

    def test_create_app_uses_existing_db(self, temp_db_path, sample_documents):
        """Test that create() uses existing database data."""
        # First, create DB with data
        from llmdoc.store import DocumentStore

        store = DocumentStore(temp_db_path, read_only=False)
        for doc in sample_documents:
            store.upsert_document(
                source_name=doc.source_name,
                source_url=doc.source_url,
                doc_url=doc.doc_url,
                title=doc.title,
                content=doc.content,
            )
        store.close()

        # Now create app
        config = Config(
            sources=[Source(name="test", url="https://example.com/llms.txt")],
            db_path=temp_db_path,
        )
        app = LLMDocApp.create(config)

        try:
            # Should have loaded documents into index
            assert app.index.document_count == len(sample_documents)
        finally:
            app.close()

    def test_create_app_empty_db(self, temp_db_path):
        """Test creating app with empty database."""
        config = Config(
            sources=[Source(name="test", url="https://example.com/llms.txt")],
            db_path=temp_db_path,
        )

        app = LLMDocApp.create(config)
        try:
            assert app.index.document_count == 0
        finally:
            app.close()

    def test_create_app_no_config(self, monkeypatch, tmp_path):
        """Test creating app without explicit config."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("LLMDOC_SOURCES", "test:https://example.com/llms.txt")
        monkeypatch.setenv("LLMDOC_DB_PATH", db_path)

        app = LLMDocApp.create()
        try:
            assert len(app.config.sources) == 1
            assert app.config.sources[0].name == "test"
        finally:
            app.close()

    def test_app_close(self, sample_config):
        """Test closing the app."""
        app = LLMDocApp.create(sample_config)
        app.close()
        # Should not raise, store should be closed

    def test_app_fetcher_concurrency(self, tmp_path):
        """Test that fetcher uses configured concurrency."""
        config = Config(
            sources=[],
            db_path=str(tmp_path / "test.db"),
            max_concurrent_fetches=10,
        )

        app = LLMDocApp.create(config)
        try:
            assert app.fetcher.max_concurrent == 10
        finally:
            app.close()

    def test_app_dataclass_fields(self, sample_config):
        """Test that app has all expected fields."""
        app = LLMDocApp.create(sample_config)

        try:
            # All fields should be accessible
            assert hasattr(app, "config")
            assert hasattr(app, "store")
            assert hasattr(app, "index")
            assert hasattr(app, "fetcher")
        finally:
            app.close()
