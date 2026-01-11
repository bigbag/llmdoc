"""BM25 indexing for document search."""

import re
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .store import Document

WORD_PATTERN = re.compile(r"\b\w+\b")
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
SENTENCE_BOUNDARIES = [".\n", ". ", "!\n", "! ", "?\n", "? "]


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find the best sentence boundary within the range [start, end).

    Searches backwards from `end` to find a sentence terminator (.!?) followed
    by whitespace. Returns the position just after the terminator if found,
    otherwise returns `end` (falling back to hard cut).

    Args:
        text: The text to search within.
        start: Start of the search range.
        end: End of the search range.

    Returns:
        Position of the best break point.
    """
    for sep in SENTENCE_BOUNDARIES:
        last_sep = text.rfind(sep, start, end)
        if last_sep > start:
            return last_sep + len(sep)
    return end


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "can",
        "will",
        "should",
        "now",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "would",
        "could",
        "ought",
        "of",
        "as",
        "how",
        "why",
        "because",
        "while",
        "also",
        "any",
        "both",
        "either",
        "neither",
        # Modal verbs
        "may",
        "might",
        "must",
        "shall",
        # Location/time
        "where",
        "until",
        "since",
        "yet",
        "still",
        "upon",
        "within",
        "without",
        "well",
        # Contraction parts (e.g., "we'll" -> ["we", "ll"])
        "ll",
        "ve",
        "re",
        "d",
        "m",
        "s",
        "t",
        "don",
        "won",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "wouldn",
    ]
)


@dataclass
class SearchResult:
    """A search result."""

    doc_url: str
    source_name: str
    source_url: str
    title: str | None
    snippet: str
    score: float


@dataclass
class DocumentChunk:
    """A chunk of a document for indexing."""

    doc_id: int | None
    doc_url: str
    source_name: str
    source_url: str
    title: str | None
    content: str
    start_pos: int


class BM25Index:
    """BM25-based search index for documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize the index.

        Args:
            chunk_size: Target size for document chunks (in characters).
            chunk_overlap: Overlap between chunks (in characters).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []
        self._lock = threading.RLock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Args:
            text: The text to tokenize.

        Returns:
            List of tokens (lowercased words).
        """
        words = WORD_PATTERN.findall(text.lower())
        return [w for w in words if w not in STOPWORDS and len(w) > 1]

    @staticmethod
    def _create_chunk(doc: Document, content: str, start_pos: int) -> DocumentChunk:
        """Create a DocumentChunk from a document and content.

        Args:
            doc: The source document.
            content: The chunk content.
            start_pos: The start position in the original document.

        Returns:
            A DocumentChunk instance.
        """
        return DocumentChunk(
            doc_id=doc.id,
            doc_url=doc.doc_url,
            source_name=doc.source_name,
            source_url=doc.source_url,
            title=doc.title,
            content=content,
            start_pos=start_pos,
        )

    def _chunk_document(self, doc: Document) -> list[DocumentChunk]:
        """Split a document into chunks for indexing.

        Uses paragraph-based chunking with fallback to character-based.
        Tracks actual positions in the original document content.

        Args:
            doc: The document to chunk.

        Returns:
            List of document chunks.
        """
        content = doc.content
        chunks: list[DocumentChunk] = []

        paragraph_positions: list[tuple[int, int]] = []
        last_end = 0

        for match in PARAGRAPH_PATTERN.finditer(content):
            if match.start() > last_end:
                paragraph_positions.append((last_end, match.start()))
            last_end = match.end()

        if last_end < len(content):
            paragraph_positions.append((last_end, len(content)))

        if not paragraph_positions and content.strip():
            paragraph_positions.append((0, len(content)))

        current_chunk = ""
        current_chunk_start = 0

        for para_start_pos, para_end_pos in paragraph_positions:
            para = content[para_start_pos:para_end_pos].strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_chunk_start = para_start_pos
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(doc, current_chunk, current_chunk_start))

                if len(para) <= self.chunk_size:
                    current_chunk = para
                    current_chunk_start = para_start_pos
                else:
                    inner_start = 0
                    while inner_start < len(para):
                        inner_end = min(inner_start + self.chunk_size, len(para))

                        if inner_end < len(para):
                            inner_end = _find_sentence_boundary(para, inner_start, inner_end)

                        chunk_text = para[inner_start:inner_end]
                        if chunk_text.strip():
                            chunk_pos = para_start_pos + inner_start
                            chunks.append(self._create_chunk(doc, chunk_text, chunk_pos))

                        next_start = inner_end - self.chunk_overlap
                        if next_start <= inner_start:
                            next_start = inner_end
                        inner_start = next_start

                    current_chunk = ""
                    current_chunk_start = 0

        if current_chunk:
            chunks.append(self._create_chunk(doc, current_chunk, current_chunk_start))

        if not chunks and content.strip():
            chunks.append(self._create_chunk(doc, content.strip(), 0))

        return chunks

    def build_index(self, documents: list[Document]) -> None:
        """Build the BM25 index from documents.

        Args:
            documents: List of documents to index.
        """
        with self._lock:
            self._chunks = []

            for doc in documents:
                self._chunks.extend(self._chunk_document(doc))

            if self._chunks:
                tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
                self._index = BM25Okapi(tokenized_corpus)
            else:
                self._index = None

    def search(self, query: str, limit: int = 5, source_filter: str | None = None) -> list[SearchResult]:
        """Search the index for relevant documents.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            source_filter: Optional source name to filter results.

        Returns:
            List of search results, sorted by relevance.
        """
        with self._lock:
            if not self._index or not self._chunks:
                return []

            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            scores = self._index.get_scores(query_tokens)

            scored_chunks = list(zip(self._chunks, scores, strict=False))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            seen_urls: set[str] = set()
            results: list[SearchResult] = []

            for chunk, score in scored_chunks:
                if score <= 0:
                    continue

                if source_filter and chunk.source_name != source_filter:
                    continue

                if chunk.doc_url not in seen_urls:
                    seen_urls.add(chunk.doc_url)

                    snippet = chunk.content[:200]
                    if len(chunk.content) > 200:
                        snippet += "..."

                    results.append(
                        SearchResult(
                            doc_url=chunk.doc_url,
                            source_name=chunk.source_name,
                            source_url=chunk.source_url,
                            title=chunk.title,
                            snippet=snippet,
                            score=float(score),
                        )
                    )

                    if len(results) >= limit:
                        break

            return results

    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        with self._lock:
            if not self._chunks:
                return 0
            return len({chunk.doc_url for chunk in self._chunks})

    @property
    def chunk_count(self) -> int:
        """Get the number of indexed chunks."""
        with self._lock:
            return len(self._chunks)

    def search_within_document(
        self,
        doc_url: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for relevant chunks within a specific document.

        Args:
            doc_url: The URL of the document to search within.
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            List of (chunk, score) tuples sorted by relevance.
        """
        with self._lock:
            if not self._index or not self._chunks:
                return []

            doc_indices = [i for i, c in enumerate(self._chunks) if c.doc_url == doc_url]
            if not doc_indices:
                return []

            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            all_scores = self._index.get_scores(query_tokens)

            scored_chunks = [(self._chunks[i], all_scores[i]) for i in doc_indices]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            return [(c, s) for c, s in scored_chunks[:top_k] if s > 0]
