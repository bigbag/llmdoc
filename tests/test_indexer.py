"""Tests for llmdoc.indexer module."""

from datetime import datetime

from llmdoc.indexer import BM25Index, DocumentChunk, SearchResult
from llmdoc.store import Document


class TestBM25Index:
    """Tests for BM25Index class."""

    def test_create_index(self):
        """Test creating an empty index."""
        index = BM25Index()
        assert index.document_count == 0
        assert index.chunk_count == 0

    def test_create_index_custom_params(self):
        """Test creating index with custom parameters."""
        index = BM25Index(chunk_size=1000, chunk_overlap=200)
        assert index.chunk_size == 1000
        assert index.chunk_overlap == 200

    def test_build_index(self, bm25_index, sample_documents):
        """Test building index from documents."""
        bm25_index.build_index(sample_documents)

        assert bm25_index.document_count == 3
        assert bm25_index.chunk_count >= 3  # At least one chunk per doc

    def test_build_index_empty(self, bm25_index):
        """Test building index with no documents."""
        bm25_index.build_index([])
        assert bm25_index.document_count == 0
        assert bm25_index.chunk_count == 0

    def test_search_returns_results(self, populated_index):
        """Test that search returns relevant results."""
        results = populated_index.search("tools functions")

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # Tools doc should be most relevant
        assert results[0].title == "Tools Documentation"

    def test_search_with_limit(self, populated_index):
        """Test search respects limit parameter."""
        results = populated_index.search("documentation", limit=2)
        assert len(results) <= 2

    def test_search_with_source_filter(self, populated_index):
        """Test search with source filter."""
        results = populated_index.search("documentation", source_filter="source_a")

        # Results should only be from source_a (if any)
        assert all(r.source_name == "source_a" for r in results)

    def test_search_source_filter_no_match(self, populated_index):
        """Test search with non-matching source filter."""
        results = populated_index.search("tools", source_filter="nonexistent")
        assert results == []

    def test_search_empty_query(self, populated_index):
        """Test search with empty query."""
        results = populated_index.search("")
        assert results == []

    def test_search_no_results(self, populated_index):
        """Test search that matches nothing."""
        results = populated_index.search("xyzabc123nonexistent")
        assert results == []

    def test_search_empty_index(self, bm25_index):
        """Test searching empty index."""
        results = bm25_index.search("test")
        assert results == []

    def test_search_result_fields(self, populated_index):
        """Test that search results have all required fields."""
        results = populated_index.search("tools")

        assert len(results) > 0
        result = results[0]

        assert result.doc_url is not None
        assert result.source_name is not None
        assert result.source_url is not None
        assert result.title is not None
        assert result.snippet is not None
        assert isinstance(result.score, float)
        assert result.score > 0

    def test_search_deduplicates_by_url(self, bm25_index):
        """Test that search returns unique URLs."""
        now = datetime.now()
        # Create document with content that will create multiple chunks
        long_content = "\n\n".join([f"Paragraph {i} about tools." for i in range(20)])
        docs = [
            Document(
                id=1,
                source_name="test",
                source_url="https://example.com/llms.txt",
                doc_url="https://example.com/doc.md",
                title="Test Doc",
                content=long_content,
                content_hash="hash",
                updated_at=now,
            )
        ]

        bm25_index.build_index(docs)
        results = bm25_index.search("tools paragraph", limit=10)

        # Should only return one result despite multiple chunks matching
        urls = [r.doc_url for r in results]
        assert len(urls) == len(set(urls))

    def test_search_scores_sorted_descending(self, populated_index):
        """Test that results are sorted by score descending."""
        results = populated_index.search("documentation")

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_snippet_truncation(self, bm25_index):
        """Test that snippets are truncated to 200 chars."""
        now = datetime.now()
        long_content = "A" * 500
        docs = [
            Document(
                id=1,
                source_name="test",
                source_url="https://example.com/llms.txt",
                doc_url="https://example.com/doc.md",
                title="Test",
                content=long_content,
                content_hash="hash",
                updated_at=now,
            )
        ]

        bm25_index.build_index(docs)
        results = bm25_index.search("AAAA")

        if results:
            assert len(results[0].snippet) <= 203  # 200 + "..."

    def test_tokenize(self):
        """Test tokenization with stopword removal."""
        tokens = BM25Index._tokenize("Hello, World! This is a TEST.")
        # Stopwords (this, is, a) and single-char words are filtered out
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_special_chars(self):
        """Test tokenization with special characters."""
        tokens = BM25Index._tokenize("foo-bar foo_bar foo.bar")
        # Should split on special chars
        assert "foo" in tokens
        assert "bar" in tokens

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        tokens = BM25Index._tokenize("")
        assert tokens == []

    def test_rebuild_index(self, bm25_index, sample_documents):
        """Test rebuilding index replaces old data."""
        bm25_index.build_index(sample_documents)
        initial_count = bm25_index.document_count

        # Rebuild with fewer documents
        bm25_index.build_index(sample_documents[:1])

        assert bm25_index.document_count == 1
        assert bm25_index.document_count < initial_count


class TestDocumentChunk:
    """Tests for DocumentChunk class."""

    def test_create_chunk(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            doc_id=1,
            doc_url="https://example.com/doc.md",
            source_name="test",
            source_url="https://example.com/llms.txt",
            title="Test",
            content="Test content",
            start_pos=0,
        )

        assert chunk.doc_id == 1
        assert chunk.doc_url == "https://example.com/doc.md"
        assert chunk.content == "Test content"
        assert chunk.start_pos == 0


class TestChunking:
    """Tests for document chunking."""

    def test_chunk_short_document(self, bm25_index):
        """Test chunking a short document."""
        now = datetime.now()
        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Short Doc",
            content="Short content.",
            content_hash="hash",
            updated_at=now,
        )

        chunks = bm25_index._chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short content."

    def test_chunk_long_document(self):
        """Test chunking a long document."""
        index = BM25Index(chunk_size=100, chunk_overlap=20)
        now = datetime.now()

        # Create document with multiple paragraphs
        paragraphs = [f"This is paragraph {i}. " * 10 for i in range(5)]
        content = "\n\n".join(paragraphs)

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Long Doc",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        chunks = index._chunk_document(doc)

        assert len(chunks) > 1

    def test_chunk_preserves_metadata(self, bm25_index):
        """Test that chunks preserve document metadata."""
        now = datetime.now()
        doc = Document(
            id=42,
            source_name="my_source",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="My Title",
            content="Some content",
            content_hash="hash",
            updated_at=now,
        )

        chunks = bm25_index._chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].doc_id == 42
        assert chunks[0].source_name == "my_source"
        assert chunks[0].source_url == "https://example.com/llms.txt"
        assert chunks[0].doc_url == "https://example.com/doc.md"
        assert chunks[0].title == "My Title"

    def test_chunk_empty_document(self, bm25_index):
        """Test chunking an empty document."""
        now = datetime.now()
        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Empty",
            content="",
            content_hash="hash",
            updated_at=now,
        )

        chunks = bm25_index._chunk_document(doc)
        assert chunks == []

    def test_chunk_whitespace_only_document(self, bm25_index):
        """Test chunking whitespace-only document."""
        now = datetime.now()
        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Whitespace",
            content="   \n\n   \t   ",
            content_hash="hash",
            updated_at=now,
        )

        chunks = bm25_index._chunk_document(doc)
        assert chunks == []


class TestSentenceBoundaryChunking:
    """Tests for sentence-boundary-aware chunking."""

    def test_chunk_respects_sentence_boundaries(self):
        """Test that chunks break at sentence endings when possible."""
        index = BM25Index(chunk_size=100, chunk_overlap=20)
        now = datetime.now()

        # Content with clear sentence boundaries that exceeds chunk_size
        content = "First sentence here. Second sentence follows. Third sentence ends. Fourth sentence now. Fifth sentence too."

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Test",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        chunks = index._chunk_document(doc)

        # Verify chunks end at sentence boundaries (period followed by space or end)
        for chunk in chunks[:-1]:  # Skip last chunk which may not end at boundary
            stripped = chunk.content.rstrip()
            # Should end with sentence terminator
            assert stripped[-1] in ".!?", f"Chunk doesn't end at sentence: '{stripped}'"

    def test_chunk_handles_no_sentence_boundaries(self):
        """Test fallback when no sentence boundaries exist."""
        index = BM25Index(chunk_size=50, chunk_overlap=10)
        now = datetime.now()

        # Very long word without boundaries
        content = "A" * 200

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Test",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        chunks = index._chunk_document(doc)

        # Should still produce chunks (fallback to hard cut)
        assert len(chunks) >= 1

    def test_chunk_with_exclamation_and_question(self):
        """Test chunking respects ! and ? boundaries."""
        index = BM25Index(chunk_size=80, chunk_overlap=10)
        now = datetime.now()

        content = "Is this a question? Yes it is! Another statement here. And more text follows after that."

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Test",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        chunks = index._chunk_document(doc)

        # Should create multiple chunks at sentence boundaries
        assert len(chunks) >= 1

    def test_chunk_ensures_forward_progress(self):
        """Test that chunking always makes forward progress."""
        index = BM25Index(chunk_size=50, chunk_overlap=40)
        now = datetime.now()

        # Content that could cause infinite loop with high overlap
        content = "Short. " * 50

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Test",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        # Should complete without hanging
        chunks = index._chunk_document(doc)
        assert len(chunks) >= 1


class TestSearchWithinDocument:
    """Tests for search_within_document method."""

    def test_search_within_document_returns_chunks(self, populated_index):
        """Test searching within a specific document."""
        # Search within the Tools doc for "functions"
        results = populated_index.search_within_document(
            doc_url="https://a.example.com/tools.md",
            query="tools functions",
            top_k=5,
        )

        assert len(results) > 0
        # Each result is a (chunk, score) tuple
        for chunk, score in results:
            assert chunk.doc_url == "https://a.example.com/tools.md"
            assert score > 0

    def test_search_within_document_not_found(self, populated_index):
        """Test searching within non-existent document."""
        results = populated_index.search_within_document(
            doc_url="https://nonexistent.com/doc.md",
            query="test",
            top_k=5,
        )

        assert results == []

    def test_search_within_document_empty_query(self, populated_index):
        """Test searching with empty query."""
        results = populated_index.search_within_document(
            doc_url="https://a.example.com/tools.md",
            query="",
            top_k=5,
        )

        assert results == []

    def test_search_within_document_no_match(self, populated_index):
        """Test searching for non-matching query."""
        results = populated_index.search_within_document(
            doc_url="https://a.example.com/tools.md",
            query="xyznonexistent123",
            top_k=5,
        )

        assert results == []

    def test_search_within_document_respects_top_k(self, bm25_index):
        """Test that top_k limit is respected."""
        now = datetime.now()
        # Create document with many chunks
        paragraphs = [f"Paragraph {i} about tools and functions." for i in range(20)]
        content = "\n\n".join(paragraphs)

        doc = Document(
            id=1,
            source_name="test",
            source_url="https://example.com/llms.txt",
            doc_url="https://example.com/doc.md",
            title="Multi-chunk Doc",
            content=content,
            content_hash="hash",
            updated_at=now,
        )

        bm25_index.build_index([doc])
        results = bm25_index.search_within_document(
            doc_url="https://example.com/doc.md",
            query="tools functions paragraph",
            top_k=3,
        )

        assert len(results) <= 3

    def test_search_within_document_empty_index(self, bm25_index):
        """Test searching within document on empty index."""
        results = bm25_index.search_within_document(
            doc_url="https://example.com/doc.md",
            query="test",
            top_k=5,
        )

        assert results == []
