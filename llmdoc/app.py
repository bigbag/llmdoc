"""Application state container for LLMDoc."""

from dataclasses import dataclass
from pathlib import Path

from .config import Config, load_config
from .fetcher import DocumentFetcher
from .indexer import BM25Index
from .store import DocumentStore


@dataclass
class LLMDocApp:
    """Encapsulates all LLMDoc application state."""

    config: Config
    store: DocumentStore
    index: BM25Index
    fetcher: DocumentFetcher

    @classmethod
    def create(cls, config: Config | None = None) -> "LLMDocApp":
        """Create and initialize the application.

        Args:
            config: Optional configuration. If not provided, loads from environment/file.

        Returns:
            Initialized LLMDocApp instance.
        """
        config = config or load_config()

        # Ensure database directory exists and initialize schema if needed
        db_path = Path(config.db_path)
        if not db_path.exists():
            init_store = DocumentStore(config.db_path, read_only=False)
            init_store.close()

        store = DocumentStore(config.db_path, read_only=True)

        # Create FTS index if enabled but missing (e.g., DB created with FTS disabled)
        if config.enable_fts and not store.has_fts_index():
            store.close()
            write_store = DocumentStore(config.db_path, read_only=False)
            write_store.create_fts_index()
            write_store.close()
            store = DocumentStore(config.db_path, read_only=True)

        index = BM25Index(store=store, enable_fts=config.enable_fts)
        existing_docs = store.get_all_documents()
        if existing_docs:
            index.build_index(existing_docs)
            index.sync_chunk_ids_from_store()

        # Initialize fetcher with concurrency limit
        fetcher = DocumentFetcher(max_concurrent=config.max_concurrent_fetches)

        return cls(config=config, store=store, index=index, fetcher=fetcher)

    def close(self) -> None:
        """Clean up resources."""
        self.store.close()

    def __enter__(self) -> "LLMDocApp":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
