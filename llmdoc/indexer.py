"""BM25 indexing for document search."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from .store import Document

if TYPE_CHECKING:
    from .store import DocumentStore

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

    id: int | None
    doc_id: int | None
    doc_url: str
    source_name: str
    source_url: str
    title: str | None
    content: str
    start_pos: int
    end_pos: int


class BM25Index:
    """BM25-based search index for documents with two-stage retrieval.

    Stage 1: DuckDB FTS with Porter stemming for broad recall
    Stage 2: Python BM25 for exact-match reranking
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        store: DocumentStore | None = None,
        enable_fts: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store = store
        self._enable_fts = enable_fts
        self._index: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []
        self._chunk_id_map: dict[int, int] = {}
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
    def _create_chunk(doc: Document, content: str, start_pos: int, end_pos: int) -> DocumentChunk:
        """Create a DocumentChunk from a document and content.

        Args:
            doc: The source document.
            content: The chunk content.
            start_pos: The start position in the original document.
            end_pos: The end position in the original document.

        Returns:
            A DocumentChunk instance.
        """
        return DocumentChunk(
            id=None,
            doc_id=doc.id,
            doc_url=doc.doc_url,
            source_name=doc.source_name,
            source_url=doc.source_url,
            title=doc.title,
            content=content,
            start_pos=start_pos,
            end_pos=end_pos,
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
        current_chunk_end = 0

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
                current_chunk_end = para_start_pos + len(para)
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(doc, current_chunk, current_chunk_start, current_chunk_end))

                if len(para) <= self.chunk_size:
                    current_chunk = para
                    current_chunk_start = para_start_pos
                    current_chunk_end = para_start_pos + len(para)
                else:
                    inner_start = 0
                    while inner_start < len(para):
                        inner_end = min(inner_start + self.chunk_size, len(para))

                        if inner_end < len(para):
                            inner_end = _find_sentence_boundary(para, inner_start, inner_end)

                        chunk_text = para[inner_start:inner_end]
                        if chunk_text.strip():
                            chunk_pos = para_start_pos + inner_start
                            chunk_end = para_start_pos + inner_end
                            chunks.append(self._create_chunk(doc, chunk_text, chunk_pos, chunk_end))

                        next_start = inner_end - self.chunk_overlap
                        if next_start <= inner_start:
                            next_start = inner_end
                        inner_start = next_start

                    current_chunk = ""
                    current_chunk_start = 0
                    current_chunk_end = 0

        if current_chunk:
            chunks.append(self._create_chunk(doc, current_chunk, current_chunk_start, current_chunk_end))

        if not chunks and content.strip():
            chunks.append(self._create_chunk(doc, content.strip(), 0, len(content.strip())))

        return chunks

    def build_index(self, documents: list[Document]) -> None:
        """Build the in-memory BM25 index from documents.

        Args:
            documents: List of documents to index.
        """
        with self._lock:
            self._chunks = []
            self._chunk_id_map = {}

            for doc in documents:
                self._chunks.extend(self._chunk_document(doc))

            if self._chunks:
                tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
                self._index = BM25Okapi(tokenized_corpus)
            else:
                self._index = None

    def sync_chunk_ids_from_store(self) -> None:
        """Sync chunk IDs from store for FTS candidate lookup."""
        if not self._store:
            return

        with self._lock:
            stored = self._store.get_all_chunks()
            self._chunk_id_map = {}

            chunk_lookup: dict[tuple[str, int, int], int] = {
                (chunk.doc_url, chunk.start_pos, chunk.end_pos): i for i, chunk in enumerate(self._chunks)
            }

            for db_chunk, doc in stored:
                key = (doc.doc_url, db_chunk.start_pos, db_chunk.end_pos)
                if key in chunk_lookup:
                    idx = chunk_lookup[key]
                    self._chunks[idx].id = db_chunk.id
                    self._chunk_id_map[db_chunk.id] = idx

    def generate_chunks_for_document(self, doc: Document) -> list[tuple[str, int, int]]:
        """Generate chunk data for a document (for storage during refresh).

        Args:
            doc: The document to generate chunks for.

        Returns:
            List of (content, start_pos, end_pos) tuples.
        """
        chunks = self._chunk_document(doc)
        return [(c.content, c.start_pos, c.end_pos) for c in chunks]

    def search(self, query: str, limit: int = 5, source_filter: str | None = None) -> list[SearchResult]:
        """Search using two-stage retrieval: FTS for recall, BM25 for reranking.

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

            candidate_indices: set[int] | None = None
            if self._enable_fts and self._store and self._chunk_id_map:
                fts_ids = self._store.get_fts_candidates(query, limit=100)
                if fts_ids:
                    candidate_indices = {self._chunk_id_map[cid] for cid in fts_ids if cid in self._chunk_id_map}

            scores = self._index.get_scores(query_tokens)

            if candidate_indices is not None:
                scored_chunks = [(self._chunks[i], scores[i]) for i in candidate_indices if scores[i] > 0]
            else:
                scored_chunks = [
                    (chunk, score) for chunk, score in zip(self._chunks, scores, strict=False) if score > 0
                ]

            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            seen_urls: set[str] = set()
            results: list[SearchResult] = []

            for chunk, score in scored_chunks:
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
