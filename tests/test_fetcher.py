"""Tests for llmdoc.fetcher module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmdoc.fetcher import DocLink, DocumentFetcher, FetchedDocument


class TestDocLink:
    """Tests for DocLink class."""

    def test_create_doc_link(self):
        """Test creating a DocLink."""
        link = DocLink(
            title="Test Doc",
            url="https://example.com/doc.md",
            description="A test document",
        )
        assert link.title == "Test Doc"
        assert link.url == "https://example.com/doc.md"
        assert link.description == "A test document"

    def test_create_doc_link_no_description(self):
        """Test creating a DocLink without description."""
        link = DocLink(title="Test", url="https://example.com/doc.md")
        assert link.description is None


class TestFetchedDocument:
    """Tests for FetchedDocument class."""

    def test_create_fetched_document(self):
        """Test creating a FetchedDocument."""
        doc = FetchedDocument(
            url="https://example.com/doc.md",
            title="Test Doc",
            content="# Test\n\nContent here.",
        )
        assert doc.url == "https://example.com/doc.md"
        assert doc.title == "Test Doc"
        assert doc.content == "# Test\n\nContent here."


class TestDocumentFetcher:
    """Tests for DocumentFetcher class."""

    def test_create_fetcher(self):
        """Test creating a fetcher."""
        fetcher = DocumentFetcher(timeout=30.0, max_concurrent=10)
        assert fetcher.timeout == 30.0
        assert fetcher.max_concurrent == 10

    def test_create_fetcher_defaults(self):
        """Test fetcher default values."""
        fetcher = DocumentFetcher()
        assert fetcher.timeout == 30.0
        assert fetcher.max_concurrent == 5


class TestParseLlmsTxt:
    """Tests for llms.txt parsing."""

    def test_parse_basic_links(self, document_fetcher, sample_llms_txt):
        """Test parsing basic links from llms.txt."""
        links = document_fetcher.parse_llms_txt(sample_llms_txt, "https://example.com/llms.txt")

        assert len(links) == 3
        assert links[0].title == "Getting Started"
        assert links[0].url == "https://example.com/getting-started.md"
        assert links[0].description == "How to get started"

    def test_parse_resolves_relative_urls(self, document_fetcher):
        """Test that relative URLs are resolved."""
        content = "- [Doc](docs/page.md)"
        links = document_fetcher.parse_llms_txt(content, "https://example.com/llms.txt")

        assert len(links) == 1
        assert links[0].url == "https://example.com/docs/page.md"

    def test_parse_absolute_urls_unchanged(self, document_fetcher):
        """Test that absolute URLs are unchanged."""
        content = "- [External](https://other.com/doc.md)"
        links = document_fetcher.parse_llms_txt(content, "https://example.com/llms.txt")

        assert len(links) == 1
        assert links[0].url == "https://other.com/doc.md"

    def test_parse_link_without_description(self, document_fetcher):
        """Test parsing links without descriptions."""
        content = "- [Just a Link](page.md)"
        links = document_fetcher.parse_llms_txt(content, "https://example.com/llms.txt")

        assert len(links) == 1
        assert links[0].title == "Just a Link"
        assert links[0].description is None

    def test_parse_empty_content(self, document_fetcher):
        """Test parsing empty content."""
        links = document_fetcher.parse_llms_txt("", "https://example.com/llms.txt")
        assert links == []

    def test_parse_no_links(self, document_fetcher):
        """Test parsing content with no links."""
        content = "# Project\n\n> Description\n\nJust text, no links."
        links = document_fetcher.parse_llms_txt(content, "https://example.com/llms.txt")
        assert links == []


class TestUrlHelpers:
    """Tests for URL helper methods."""

    def test_is_markdown_url(self, document_fetcher):
        """Test detecting markdown URLs."""
        assert document_fetcher._is_markdown_url("https://example.com/doc.md")
        assert document_fetcher._is_markdown_url("https://example.com/doc.markdown")
        assert not document_fetcher._is_markdown_url("https://example.com/doc.html")
        assert not document_fetcher._is_markdown_url("https://example.com/doc.txt")

    def test_is_text_url(self, document_fetcher):
        """Test detecting text URLs."""
        assert document_fetcher._is_text_url("https://example.com/doc.txt")
        assert document_fetcher._is_text_url("https://example.com/llms.txt")
        assert not document_fetcher._is_text_url("https://example.com/doc.md")
        assert not document_fetcher._is_text_url("https://example.com/doc.html")

    def test_is_llms_txt_url(self, document_fetcher):
        """Test detecting llms.txt URLs."""
        assert document_fetcher.is_llms_txt_url("https://example.com/llms.txt")
        assert document_fetcher.is_llms_txt_url("https://example.com/path/llms.txt")
        assert not document_fetcher.is_llms_txt_url("https://example.com/other.txt")
        assert not document_fetcher.is_llms_txt_url("https://example.com/llms.md")

    def test_is_html_content(self, document_fetcher):
        """Test detecting HTML content."""
        assert document_fetcher._is_html("<!DOCTYPE html><html>...")
        assert document_fetcher._is_html("<html><head>...")
        assert document_fetcher._is_html("<body>Content</body>")
        assert not document_fetcher._is_html("# Markdown\n\nContent")
        assert not document_fetcher._is_html("Plain text")


class TestTitleExtraction:
    """Tests for title extraction."""

    def test_extract_title_from_markdown(self, document_fetcher, sample_markdown):
        """Test extracting title from markdown."""
        title = document_fetcher._extract_title_from_markdown(sample_markdown)
        assert title == "Getting Started"

    def test_extract_title_no_h1(self, document_fetcher):
        """Test extracting title when no H1 exists."""
        content = "## Second Level\n\nNo H1 heading."
        title = document_fetcher._extract_title_from_markdown(content)
        assert title is None

    def test_extract_title_h1_not_at_start(self, document_fetcher):
        """Test extracting title when H1 is not at the start."""
        content = "Some intro text\n\n# Title Here\n\nMore content."
        title = document_fetcher._extract_title_from_markdown(content)
        assert title == "Title Here"


class TestHtmlToMarkdown:
    """Tests for HTML to Markdown conversion."""

    def test_convert_simple_html(self, document_fetcher):
        """Test converting simple HTML to markdown."""
        html = "<h1>Title</h1><p>Paragraph text.</p>"
        md = document_fetcher._convert_html_to_markdown(html)

        assert "Title" in md
        assert "Paragraph text" in md

    def test_convert_strips_script_tags(self, document_fetcher):
        """Test that script tags are converted (markdownify strips tags but may keep text)."""
        html = "<p>Text</p><script>alert('evil')</script>"
        md = document_fetcher._convert_html_to_markdown(html)

        assert "Text" in md
        # Script tag itself should be removed
        assert "<script>" not in md


class TestFetchUrl:
    """Tests for URL fetching."""

    @pytest.mark.asyncio
    async def test_fetch_document_markdown(self, document_fetcher):
        """Test fetching a markdown document."""
        mock_response = MagicMock()
        mock_response.text = "# Title\n\nContent here."
        mock_response.headers = {"content-type": "text/markdown"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            doc = await document_fetcher.fetch_document("https://example.com/doc.md")

            assert doc.url == "https://example.com/doc.md"
            assert doc.title == "Title"
            assert "Content here" in doc.content

    @pytest.mark.asyncio
    async def test_fetch_document_html_converted(self, document_fetcher):
        """Test that HTML is converted to markdown."""
        mock_response = MagicMock()
        mock_response.text = "<h1>Title</h1><p>Content</p>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            doc = await document_fetcher.fetch_document("https://example.com/page.html")

            assert "Title" in doc.content
            assert "<h1>" not in doc.content


class TestFetchAllFromSource:
    """Tests for fetching all documents from a source."""

    @pytest.mark.asyncio
    async def test_fetch_all_from_llms_txt(self, document_fetcher):
        """Test fetching all documents from an llms.txt source."""
        llms_txt_content = """# Project
- [Doc One](doc1.md)
- [Doc Two](doc2.md)
"""
        doc1_content = "# Doc One\n\nContent one."
        doc2_content = "# Doc Two\n\nContent two."

        async def mock_get(url):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-type": "text/plain"}

            if url == "https://example.com/llms.txt":
                mock_response.text = llms_txt_content
            elif url == "https://example.com/doc1.md":
                mock_response.text = doc1_content
            elif url == "https://example.com/doc2.md":
                mock_response.text = doc2_content
            else:
                raise httpx.HTTPError(f"Not found: {url}")

            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            docs, errors = await document_fetcher.fetch_all_from_source("https://example.com/llms.txt")

            assert len(docs) == 2
            assert len(errors) == 0
            assert docs[0].title == "Doc One"
            assert docs[1].title == "Doc Two"

    @pytest.mark.asyncio
    async def test_fetch_all_direct_document(self, document_fetcher):
        """Test fetching a direct document (not llms.txt)."""
        doc_content = "# Direct Doc\n\nContent."

        mock_response = MagicMock()
        mock_response.text = doc_content
        mock_response.headers = {"content-type": "text/markdown"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            docs, errors = await document_fetcher.fetch_all_from_source("https://example.com/doc.md")

            assert len(docs) == 1
            assert docs[0].title == "Direct Doc"
            assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_fetch_all_partial_failure(self, document_fetcher):
        """Test that partial failures are reported."""
        llms_txt_content = """# Project
- [Good Doc](good.md)
- [Bad Doc](bad.md)
"""

        async def mock_get(url):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-type": "text/plain"}

            if url == "https://example.com/llms.txt":
                mock_response.text = llms_txt_content
            elif url == "https://example.com/good.md":
                mock_response.text = "# Good\n\nContent."
            elif url == "https://example.com/bad.md":
                raise httpx.HTTPError("Server error")

            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            docs, errors = await document_fetcher.fetch_all_from_source("https://example.com/llms.txt")

            assert len(docs) == 1
            assert docs[0].title == "Good"
            assert len(errors) == 1
            assert "bad.md" in errors[0]
