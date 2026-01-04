"""Tests for llmdoc.store module."""

from llmdoc.store import Document, DocumentStore


class TestDocumentStore:
    """Tests for DocumentStore class."""

    def test_create_store(self, temp_db_path):
        """Test creating a new document store."""
        store = DocumentStore(temp_db_path, read_only=False)
        assert store.db_path == temp_db_path
        assert not store.read_only
        store.close()

    def test_create_read_only_store(self, temp_db_path):
        """Test creating a read-only store."""
        # First create the DB
        store = DocumentStore(temp_db_path, read_only=False)
        store.close()

        # Now open read-only
        store = DocumentStore(temp_db_path, read_only=True)
        assert store.read_only
        store.close()

    def test_upsert_new_document(self, document_store):
        """Test inserting a new document."""
        doc = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Test Doc",
            content="Test content",
        )

        assert doc.source_name == "test"
        assert doc.doc_url == "https://example.com/doc.md"
        assert doc.title == "Test Doc"
        assert doc.content == "Test content"
        assert doc.id is not None

    def test_upsert_updates_existing_document(self, document_store):
        """Test updating an existing document."""
        # Insert
        doc1 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Original Title",
            content="Original content",
        )

        # Update with new content
        doc2 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Updated Title",
            content="Updated content",
        )

        assert doc1.id == doc2.id  # Same document
        assert doc2.title == "Updated Title"
        assert doc2.content == "Updated content"

    def test_upsert_unchanged_document(self, document_store):
        """Test upserting a document with unchanged content."""
        # Insert
        doc1 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Title",
            content="Same content",
        )

        # Upsert again with same content
        doc2 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Title",
            content="Same content",
        )

        assert doc1.id == doc2.id
        # Both should have the same document ID since content is unchanged

    def test_get_all_documents(self, populated_store):
        """Test getting all documents."""
        docs = populated_store.get_all_documents()
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_get_document_by_url(self, populated_store, sample_documents):
        """Test getting a document by URL."""
        doc = populated_store.get_document_by_url("https://a.example.com/tools.md")
        assert doc is not None
        assert doc.title == "Tools Documentation"

    def test_get_document_by_url_not_found(self, populated_store):
        """Test getting a non-existent document."""
        doc = populated_store.get_document_by_url("https://nonexistent.com/doc.md")
        assert doc is None

    def test_delete_stale_documents(self, populated_store):
        """Test deleting stale documents."""
        # Keep only one URL as valid
        valid_urls = {"https://a.example.com/tools.md"}
        deleted = populated_store.delete_stale_documents("source_a", valid_urls)

        assert deleted == 1  # resources.md was deleted

        # Verify
        all_docs = populated_store.get_all_documents()
        remaining = [d for d in all_docs if d.source_name == "source_a"]
        assert len(remaining) == 1
        assert remaining[0].doc_url == "https://a.example.com/tools.md"

    def test_delete_stale_documents_empty_valid_urls(self, populated_store):
        """Test deleting all documents when valid_urls is empty."""
        # Get count before deletion
        all_docs = populated_store.get_all_documents()
        before = [d for d in all_docs if d.source_name == "source_a"]
        assert len(before) > 0  # Ensure we have docs to delete

        populated_store.delete_stale_documents("source_a", set())

        # Should delete all docs from source_a
        all_docs = populated_store.get_all_documents()
        remaining = [d for d in all_docs if d.source_name == "source_a"]
        assert len(remaining) == 0

    def test_delete_stale_documents_all_valid(self, populated_store):
        """Test that no documents are deleted when all are valid."""
        valid_urls = {
            "https://a.example.com/tools.md",
            "https://a.example.com/resources.md",
        }
        deleted = populated_store.delete_stale_documents("source_a", valid_urls)

        assert deleted == 0

    def test_get_source_stats(self, populated_store):
        """Test getting source statistics."""
        stats = populated_store.get_source_stats()

        assert len(stats) == 2

        stats_by_name = {s["name"]: s for s in stats}
        assert stats_by_name["source_a"]["doc_count"] == 2
        assert stats_by_name["source_b"]["doc_count"] == 1

    def test_get_source_stats_empty_store(self, document_store):
        """Test getting stats from empty store."""
        stats = document_store.get_source_stats()
        assert stats == []

    def test_content_hash_changes_on_update(self, document_store):
        """Test that content hash changes when content changes."""
        doc1 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Title",
            content="Content A",
        )

        doc2 = document_store.upsert_document(
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Title",
            content="Content B",
        )

        assert doc1.content_hash != doc2.content_hash

    def test_close_store(self, temp_db_path):
        """Test closing the store."""
        store = DocumentStore(temp_db_path, read_only=False)
        store.close()
        # Should not raise

    def test_multiple_sources_same_url(self, document_store):
        """Test that doc_url is unique regardless of source."""
        doc1 = document_store.upsert_document(
            source_name="source_a",
            source_url="https://a.example.com/llms.txt",
            doc_url="https://shared.com/doc.md",
            title="Title A",
            content="Content A",
        )

        # Same doc_url from different source should update, not create new
        doc2 = document_store.upsert_document(
            source_name="source_b",
            source_url="https://b.example.com/llms.txt",
            doc_url="https://shared.com/doc.md",
            title="Title B",
            content="Content B",
        )

        assert doc1.id == doc2.id
        assert doc2.source_name == "source_b"  # Updated to new source

    def test_context_manager(self, temp_db_path):
        """Test using DocumentStore as context manager."""
        with DocumentStore(temp_db_path, read_only=False) as store:
            store.upsert_document(
                source_name="test",
                source_url="https://example.com/llms.txt",
                doc_url="https://example.com/doc.md",
                title="Test",
                content="Content",
            )
        # Connection should be closed after exiting context
        # Re-open to verify data was persisted
        with DocumentStore(temp_db_path, read_only=True) as store:
            doc = store.get_document_by_url("https://example.com/doc.md")
            assert doc is not None
            assert doc.title == "Test"
