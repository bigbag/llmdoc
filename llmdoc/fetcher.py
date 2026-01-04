"""Document fetching logic for llms.txt and markdown files."""

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from markdownify import markdownify


@dataclass
class DocLink:
    """A link extracted from llms.txt."""

    title: str
    url: str
    description: str | None = None


@dataclass
class FetchedDocument:
    """A fetched document."""

    url: str
    title: str | None
    content: str


class DocumentFetcher:
    """Fetches documents from URLs and llms.txt files."""

    def __init__(self, timeout: float = 30.0, max_concurrent: int = 5) -> None:
        """Initialize the fetcher.

        Args:
            timeout: HTTP request timeout in seconds.
            max_concurrent: Maximum number of concurrent document fetches.
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None

    async def fetch_url(self, url: str) -> str:
        """Fetch content from a URL.

        Args:
            url: The URL to fetch.

        Returns:
            The content as a string.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def parse_llms_txt(self, content: str, base_url: str) -> list[DocLink]:
        """Parse an llms.txt file and extract document links.

        The llms.txt format uses markdown with:
        - H1 for project name
        - Blockquote for summary
        - H2 sections with lists of links

        Args:
            content: The llms.txt content.
            base_url: The base URL for resolving relative links.

        Returns:
            List of document links.
        """
        links: list[DocLink] = []

        # Match markdown links: [title](url) or [title](url): description
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)(?:\s*:\s*(.+?))?(?:\n|$)")

        for match in link_pattern.finditer(content):
            title = match.group(1).strip()
            url = match.group(2).strip()
            description = match.group(3).strip() if match.group(3) else None

            # Resolve relative URLs
            absolute_url = urljoin(base_url, url)

            links.append(DocLink(title=title, url=absolute_url, description=description))

        return links

    def _is_markdown_url(self, url: str) -> bool:
        """Check if URL points to a markdown file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith(".md") or path.endswith(".markdown")

    def _is_text_url(self, url: str) -> bool:
        """Check if URL points to a plain text file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith(".txt")

    def _is_html(self, content: str) -> bool:
        """Check if content appears to be HTML based on content inspection."""
        # Simple heuristic: check for common HTML tags
        return bool(re.search(r"<(!DOCTYPE|html|head|body)", content, re.IGNORECASE))

    def _extract_title_from_markdown(self, content: str) -> str | None:
        """Extract the first H1 title from markdown content."""
        match = re.search(r"^#\s+(.+?)$", content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _convert_html_to_markdown(self, html: str) -> str:
        """Convert HTML content to markdown."""
        return markdownify(html, heading_style="ATX", strip=["script", "style"])

    async def fetch_document(self, url: str) -> FetchedDocument:
        """Fetch a document and convert to markdown if needed.

        Args:
            url: The document URL.

        Returns:
            The fetched document.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.text
            content_type = response.headers.get("content-type", "")

            if self._is_markdown_url(url) or self._is_text_url(url):
                pass
            elif "text/markdown" in content_type:
                pass
            elif "text/html" in content_type:
                content = self._convert_html_to_markdown(content)
            elif self._is_html(content):
                content = self._convert_html_to_markdown(content)

            # Extract title
            title = self._extract_title_from_markdown(content)

            return FetchedDocument(url=url, title=title, content=content)

    async def _fetch_with_limit(self, url: str) -> FetchedDocument:
        """Fetch a document with concurrency limiting.

        Args:
            url: The document URL.

        Returns:
            The fetched document.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async with self._semaphore:
            return await self.fetch_document(url)

    async def fetch_llms_txt(self, url: str) -> list[DocLink]:
        """Fetch and parse an llms.txt file.

        Args:
            url: The URL of the llms.txt file.

        Returns:
            List of document links from the llms.txt.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        content = await self.fetch_url(url)
        return self.parse_llms_txt(content, url)

    def is_llms_txt_url(self, url: str) -> bool:
        """Check if a URL points to an llms.txt file."""
        parsed = urlparse(url)
        return parsed.path.endswith("llms.txt") or parsed.path.endswith("/llms.txt")

    async def fetch_all_from_source(self, source_url: str) -> tuple[list[FetchedDocument], list[str]]:
        """Fetch all documents from a source URL.

        If the URL is an llms.txt file, fetch all linked documents.
        Otherwise, fetch the URL directly as a document.

        Documents are fetched concurrently with a limit of max_concurrent
        simultaneous requests to avoid overwhelming target servers.

        Args:
            source_url: The source URL (llms.txt or direct document).

        Returns:
            Tuple of (list of fetched documents, list of error messages).
        """
        documents: list[FetchedDocument] = []
        errors: list[str] = []

        try:
            if self.is_llms_txt_url(source_url):
                # Fetch llms.txt and then all linked documents
                links = await self.fetch_llms_txt(source_url)

                # Fetch documents concurrently with semaphore limiting
                tasks = [self._fetch_with_limit(link.url) for link in links]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for link, result in zip(links, results):
                    if isinstance(result, BaseException):
                        errors.append(f"Failed to fetch {link.url}: {result}")
                    else:
                        # Use the link title if document doesn't have one
                        if not result.title:
                            result = FetchedDocument(
                                url=result.url,
                                title=link.title,
                                content=result.content,
                            )
                        documents.append(result)
            else:
                # Fetch as a direct document
                doc = await self.fetch_document(source_url)
                documents.append(doc)

        except Exception as e:
            errors.append(f"Failed to fetch source {source_url}: {e}")

        return documents, errors
