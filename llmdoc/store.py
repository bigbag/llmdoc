"""DuckDB storage layer for documents."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb


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
        self.conn = duckdb.connect(db_path, read_only=read_only)
        if not read_only:
            self._init_schema()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        # Check if table exists using DuckDB syntax
        tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'documents'"
        ).fetchall()

        if tables:
            # Table exists, check for migrations
            columns = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
            ).fetchall()
            column_names = [col[0] for col in columns]

            if "source_name" not in column_names:
                # Add source_name column with default value from source_url
                self.conn.execute("ALTER TABLE documents ADD COLUMN source_name TEXT DEFAULT ''")
                # Update existing rows to extract name from URL
                self.conn.execute(
                    """
                    UPDATE documents
                    SET source_name = REPLACE(REPLACE(
                        REGEXP_EXTRACT(source_url, '://([^/]+)', 1),
                        '.', '_'), '-', '_')
                    WHERE source_name = '' OR source_name IS NULL
                """
                )

            # Remove fetched_at column if it exists (no longer needed)
            if "fetched_at" in column_names:
                self.conn.execute("ALTER TABLE documents DROP COLUMN fetched_at")
        else:
            # Create sequence for auto-increment
            self.conn.execute(
                """
                CREATE SEQUENCE IF NOT EXISTS documents_id_seq
            """
            )
            # Create new table with source_name
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
                    source_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    doc_url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """
            )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source_url ON documents(source_url)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source_name ON documents(source_name)
        """
        )
        self.conn.commit()

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute a hash of the content for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

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

        # Check if document exists
        existing = self.conn.execute(
            "SELECT id, content_hash FROM documents WHERE doc_url = ?",
            [doc_url],
        ).fetchone()

        if existing:
            doc_id, old_hash = existing
            if old_hash != content_hash:
                # Content changed, update
                self.conn.execute(
                    """
                    UPDATE documents
                    SET source_name = ?, source_url = ?, title = ?, content = ?,
                        content_hash = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    [source_name, source_url, title, content, content_hash, now, doc_id],
                )
                self.conn.commit()
            else:
                # Content unchanged, but update timestamp to track last fetch
                self.conn.execute(
                    "UPDATE documents SET updated_at = ? WHERE id = ?",
                    [now, doc_id],
                )
                self.conn.commit()
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
            self.conn.execute(
                """
                INSERT INTO documents (source_name, source_url, doc_url, title, content,
                                       content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [source_name, source_url, doc_url, title, content, content_hash, now],
            )
            self.conn.commit()
            # Get the inserted ID
            result = self.conn.execute("SELECT id FROM documents WHERE doc_url = ?", [doc_url]).fetchone()
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
        rows = self.conn.execute(
            """
            SELECT id, source_name, source_url, doc_url, title, content,
                   content_hash, updated_at
            FROM documents
            """
        ).fetchall()

        return [
            Document(
                id=row[0],
                source_name=row[1],
                source_url=row[2],
                doc_url=row[3],
                title=row[4],
                content=row[5],
                content_hash=row[6],
                updated_at=row[7],
            )
            for row in rows
        ]

    def get_document_by_url(self, doc_url: str) -> Document | None:
        """Get a document by its URL.

        Args:
            doc_url: The URL of the document.

        Returns:
            The document if found, None otherwise.
        """
        row = self.conn.execute(
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

    def delete_stale_documents(self, source_name: str, valid_urls: set[str]) -> int:
        """Delete documents from a source that are no longer valid.

        Args:
            source_name: The source name.
            valid_urls: Set of URLs that are still valid.

        Returns:
            Number of deleted documents.
        """
        if not valid_urls:
            # Delete all documents from this source
            result = self.conn.execute("DELETE FROM documents WHERE source_name = ?", [source_name])
            self.conn.commit()
            return result.rowcount if hasattr(result, "rowcount") else 0

        # Get current doc URLs for this source
        current_urls = self.conn.execute(
            "SELECT doc_url FROM documents WHERE source_name = ?", [source_name]
        ).fetchall()

        stale_urls = [url[0] for url in current_urls if url[0] not in valid_urls]

        if stale_urls:
            placeholders = ",".join(["?" for _ in stale_urls])
            self.conn.execute(f"DELETE FROM documents WHERE doc_url IN ({placeholders})", stale_urls)
            self.conn.commit()
            return len(stale_urls)

        return 0

    def get_source_stats(self) -> list[dict]:
        """Get statistics for each source."""
        rows = self.conn.execute(
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
        self.conn.close()

    def __enter__(self) -> "DocumentStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
