"""DuckDB storage layer for documents."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection


@dataclass
class Document:
    """A stored document."""

    id: int | None
    source_name: str
    source_url: str
    doc_url: str
    title: str | None
    content: str
    content_hash: str
    updated_at: datetime


class DocumentStore:
    """DuckDB-based document storage."""

    def __init__(self, db_path: str, read_only: bool = False) -> None:
        """Initialize the document store.

        Args:
            db_path: Path to the DuckDB database file.
            read_only: If True, open in read-only mode (allows multiple instances).
        """
        self.db_path = db_path
        self.read_only = read_only
        self._ensure_db_dir()
        self.conn: DuckDBPyConnection | None = None
        self._connect()

        if not read_only:
            self._init_schema()

    def _connect(self) -> None:
        """Establish connection to the database."""
        self.conn = duckdb.connect(self.db_path, read_only=self.read_only)

    def _ensure_connected(self) -> DuckDBPyConnection:
        """Ensure connection is open and valid."""
        try:
            if self.conn:
                self.conn.execute("SELECT 1")
                return self.conn
        except (duckdb.ConnectionException, AttributeError, Exception):
            pass

        # Reconnect on any failure
        self._connect()
        if self.conn is None:
            raise RuntimeError("Failed to connect to database")
        return self.conn

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        conn = self._ensure_connected()
        # Create sequence for auto-incrementing id
        conn.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
                source_name TEXT,
                source_url TEXT,
                doc_url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                content_hash TEXT,
                updated_at TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_url ON documents(doc_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_name ON documents(source_name)")

        table_info = conn.execute("PRAGMA table_info(documents)").fetchall()
        column_names = [info[1] for info in table_info]

        if "source_name" not in column_names:
            # Add source_name column with default value from source_url
            conn.execute("ALTER TABLE documents ADD COLUMN source_name TEXT DEFAULT ''")
            # Update existing rows to extract name from URL
            conn.execute(
                """
                UPDATE documents
                SET source_name = CASE
                    WHEN source_url LIKE 'http%' THEN
                        regexp_replace(source_url, '^https?://([^/]+).*$', '\\1')
                    ELSE 'unknown'
                END
                WHERE source_name = ''
                """
            )

        # Remove fetched_at column if it exists (no longer needed)
        if "fetched_at" in column_names:
            conn.execute("ALTER TABLE documents DROP COLUMN fetched_at")

        conn.commit()

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute a hash of the content for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_document(row: tuple) -> Document:
        """Convert a database row to a Document.

        Args:
            row: Database row tuple (id, source_name, source_url, doc_url,
                 title, content, content_hash, updated_at).

        Returns:
            Document instance.
        """
        return Document(
            id=row[0],
            source_name=row[1],
            source_url=row[2],
            doc_url=row[3],
            title=row[4],
            content=row[5],
            content_hash=row[6],
            updated_at=row[7],
        )

    def upsert_document(
        self,
        source_name: str,
        source_url: str,
        doc_url: str,
        title: str | None,
        content: str,
    ) -> Document:
        """Insert or update a document.

        Args:
            source_name: The name of the source (e.g., 'fast_mcp').
            source_url: The URL of the source (llms.txt or config).
            doc_url: The URL of the actual document.
            title: The document title.
            content: The document content.

        Returns:
            The upserted document.
        """
        content_hash = self._compute_hash(content)
        now = datetime.now()

        conn = self._ensure_connected()
        # Check if document exists
        existing = conn.execute(
            "SELECT id, content_hash FROM documents WHERE doc_url = ?",
            [doc_url],
        ).fetchone()

        if existing:
            doc_id, old_hash = existing
            if old_hash != content_hash:
                # Content changed, update
                conn.execute(
                    """
                    UPDATE documents
                    SET source_name = ?, source_url = ?, title = ?, content = ?,
                        content_hash = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    [
                        source_name,
                        source_url,
                        title,
                        content,
                        content_hash,
                        now,
                        doc_id,
                    ],
                )
            else:
                # Content unchanged, but update timestamp to track last fetch
                conn.execute(
                    "UPDATE documents SET updated_at = ? WHERE id = ?",
                    [now, doc_id],
                )
            conn.commit()
            return Document(
                id=doc_id,
                source_name=source_name,
                source_url=source_url,
                doc_url=doc_url,
                title=title,
                content=content,
                content_hash=content_hash,
                updated_at=now,
            )
        else:
            # Insert new document
            conn.execute(
                """
                INSERT INTO documents (source_name, source_url, doc_url, title, content,
                                       content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [source_name, source_url, doc_url, title, content, content_hash, now],
            )
            conn.commit()
            # Get the inserted ID
            result = conn.execute("SELECT id FROM documents WHERE doc_url = ?", [doc_url]).fetchone()
            doc_id = result[0] if result else None

            return Document(
                id=doc_id,
                source_name=source_name,
                source_url=source_url,
                doc_url=doc_url,
                title=title,
                content=content,
                content_hash=content_hash,
                updated_at=now,
            )

    def get_all_documents(self) -> list[Document]:
        """Get all documents from the store."""
        conn = self._ensure_connected()
        rows = conn.execute(
            """
            SELECT id, source_name, source_url, doc_url, title, content,
                   content_hash, updated_at
            FROM documents
            """
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def get_document_by_url(self, doc_url: str) -> Document | None:
        """Get a document by its URL.

        Args:
            doc_url: The URL of the document.

        Returns:
            The document if found, None otherwise.
        """
        conn = self._ensure_connected()
        row = conn.execute(
            """
            SELECT id, source_name, source_url, doc_url, title, content,
                   content_hash, updated_at
            FROM documents
            WHERE doc_url = ?
            """,
            [doc_url],
        ).fetchone()

        if not row:
            return None

        return self._row_to_document(row)

    def delete_stale_documents(self, source_name: str, valid_urls: set[str]) -> int:
        """Delete documents from a source that are no longer valid.

        Args:
            source_name: The source name.
            valid_urls: Set of URLs that are still valid.

        Returns:
            Number of deleted documents.
        """
        conn = self._ensure_connected()
        if not valid_urls:
            # Delete all documents from this source
            result = conn.execute("DELETE FROM documents WHERE source_name = ?", [source_name])
            conn.commit()
            return result.rowcount if hasattr(result, "rowcount") else 0

        # Get current doc URLs for this source
        current_urls = conn.execute("SELECT doc_url FROM documents WHERE source_name = ?", [source_name]).fetchall()

        stale_urls = [url[0] for url in current_urls if url[0] not in valid_urls]

        if stale_urls:
            placeholders = ",".join(["?" for _ in stale_urls])
            conn.execute(f"DELETE FROM documents WHERE doc_url IN ({placeholders})", stale_urls)
            conn.commit()
            return len(stale_urls)

        return 0

    def get_source_stats(self) -> list[dict]:
        """Get statistics for each source."""
        conn = self._ensure_connected()
        rows = conn.execute(
            """
            SELECT source_name, source_url, COUNT(*) as doc_count, MAX(updated_at) as last_updated
            FROM documents
            GROUP BY source_name, source_url
            """
        ).fetchall()

        return [
            {
                "name": row[0],
                "url": row[1],
                "doc_count": row[2],
                "last_updated": row[3],
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "DocumentStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
