"""SQLite persistence repository for document metadata, sessions, and canonical turns."""
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from ..domain.enums import DocumentStatus, EvidenceQuality, GroundingStatus
from ..domain.models import Document, Session, Turn, ClaimSupport, ToolExecutionRecord, EvidenceChunk, Chunk, Score


class SQLiteRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    collection TEXT DEFAULT 'default',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    raw_answer TEXT NOT NULL,
                    canonical_answer TEXT NOT NULL,
                    grounding_status TEXT NOT NULL,
                    tools_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    invalid_citations_json TEXT NOT NULL,
                    claims_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    evidence_quality TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_docs_collection ON documents(collection);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """)

    # ------------------ Document Operations ------------------ #
    def save_document(self, doc: Document) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, content_type, sha256, size_bytes,
                    status, chunk_count, collection, error_message,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename=excluded.filename,
                    content_type=excluded.content_type,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    collection=excluded.collection,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json;
                """,
                (
                    doc.document_id,
                    doc.filename,
                    doc.content_type,
                    doc.sha256,
                    doc.size_bytes,
                    doc.status.value if isinstance(doc.status, DocumentStatus) else str(doc.status),
                    doc.chunk_count,
                    doc.collection,
                    doc.error_message,
                    doc.created_at,
                    doc.updated_at,
                    json.dumps(doc.metadata, ensure_ascii=False),
                ),
            )

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            if not row:
                return None
            return Document(
                document_id=row["document_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                status=DocumentStatus(row["status"]),
                chunk_count=row["chunk_count"],
                collection=row["collection"],
                error_message=row["error_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )

    def get_document_by_sha256(self, sha256: str, collection: str = "default") -> Optional[Document]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ? AND collection = ?", (sha256, collection)
            ).fetchone()
            if not row:
                return None
            return Document(
                document_id=row["document_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                status=DocumentStatus(row["status"]),
                chunk_count=row["chunk_count"],
                collection=row["collection"],
                error_message=row["error_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )

    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Persist chunks for a document into SQLite."""
        if not chunks:
            return
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, ordinal, text, section,
                    token_count, sha256, evidence_quality, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.ordinal,
                        c.text,
                        c.section,
                        c.token_count,
                        c.sha256,
                        c.evidence_quality.value if isinstance(c.evidence_quality, EvidenceQuality) else str(c.evidence_quality),
                        json.dumps(c.metadata, ensure_ascii=False),
                    )
                    for c in chunks
                ],
            )

    def get_all_chunks(self) -> List[Chunk]:
        """Retrieve all stored chunks across all documents."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY document_id, ordinal ASC").fetchall()
            return [
                Chunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    ordinal=r["ordinal"],
                    text=r["text"],
                    section=r["section"],
                    token_count=r["token_count"],
                    sha256=r["sha256"],
                    evidence_quality=EvidenceQuality(r["evidence_quality"]),
                    metadata=json.loads(r["metadata_json"] or "{}"),
                )
                for r in rows
            ]

    def get_chunks_for_document(self, document_id: str) -> List[Chunk]:
        """Retrieve chunks for a single document."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal ASC", (document_id,)
            ).fetchall()
            return [
                Chunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    ordinal=r["ordinal"],
                    text=r["text"],
                    section=r["section"],
                    token_count=r["token_count"],
                    sha256=r["sha256"],
                    evidence_quality=EvidenceQuality(r["evidence_quality"]),
                    metadata=json.loads(r["metadata_json"] or "{}"),
                )
                for r in rows
            ]

    def delete_chunks_for_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            return cur.rowcount

    def list_documents(self, collection: Optional[str] = None) -> List[Document]:
        with self._get_connection() as conn:
            query = "SELECT * FROM documents"
            params = ()
            if collection:
                query += " WHERE collection = ?"
                params = (collection,)
            query += " ORDER BY created_at ASC"
            rows = conn.execute(query, params).fetchall()
            return [
                Document(
                    document_id=r["document_id"],
                    filename=r["filename"],
                    content_type=r["content_type"],
                    sha256=r["sha256"],
                    size_bytes=r["size_bytes"],
                    status=DocumentStatus(r["status"]),
                    chunk_count=r["chunk_count"],
                    collection=r["collection"],
                    error_message=r["error_message"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    metadata=json.loads(r["metadata_json"] or "{}"),
                )
                for r in rows
            ]

    def delete_document(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return cur.rowcount > 0

    # ------------------ Session & Turn Operations ------------------ #
    def create_session(self, session_id: str, title: str = "New Conversation") -> Session:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(session_id) DO UPDATE SET updated_at = datetime('now');
                """,
                (session_id, title),
            )
        return self.get_session(session_id) or Session(session_id=session_id, title=title)

    def save_canonical_turn(self, turn: Turn) -> None:
        """Store or update the authoritative assistant turn to prevent state corruption."""
        with self._get_connection() as conn:
            # Ensure session exists
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?);
                """,
                (turn.session_id, f"Session {turn.session_id[:8]}", turn.created_at, turn.created_at),
            )
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, user_input, raw_answer, canonical_answer,
                    grounding_status, tools_json, citations_json, invalid_citations_json,
                    claims_json, evidence_json, created_at, runtime_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    raw_answer=excluded.raw_answer,
                    canonical_answer=excluded.canonical_answer,
                    grounding_status=excluded.grounding_status,
                    tools_json=excluded.tools_json,
                    citations_json=excluded.citations_json,
                    invalid_citations_json=excluded.invalid_citations_json,
                    claims_json=excluded.claims_json,
                    evidence_json=excluded.evidence_json,
                    runtime_seconds=excluded.runtime_seconds;
                """,
                (
                    turn.turn_id,
                    turn.session_id,
                    turn.user_input,
                    turn.raw_answer,
                    turn.canonical_answer,
                    turn.grounding_status.value if isinstance(turn.grounding_status, GroundingStatus) else str(turn.grounding_status),
                    json.dumps([t.to_dict() for t in turn.tool_executions], ensure_ascii=False),
                    json.dumps(turn.citations, ensure_ascii=False),
                    json.dumps(turn.invalid_citations, ensure_ascii=False),
                    json.dumps([c.to_dict() for c in turn.claims], ensure_ascii=False),
                    json.dumps([e.to_dict() for e in turn.evidence], ensure_ascii=False),
                    turn.created_at,
                    turn.runtime_seconds,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?", (turn.session_id,))

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._get_connection() as conn:
            s_row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not s_row:
                return None
            
            t_rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC", (session_id,)
            ).fetchall()
            
            turns = []
            for r in t_rows:
                tools = [ToolExecutionRecord(**t) for t in json.loads(r["tools_json"])]
                claims = [ClaimSupport(**c) for c in json.loads(r["claims_json"])]
                turns.append(
                    Turn(
                        turn_id=r["turn_id"],
                        session_id=r["session_id"],
                        user_input=r["user_input"],
                        raw_answer=r["raw_answer"],
                        canonical_answer=r["canonical_answer"],
                        grounding_status=GroundingStatus(r["grounding_status"]),
                        tool_executions=tools,
                        citations=json.loads(r["citations_json"]),
                        invalid_citations=json.loads(r["invalid_citations_json"]),
                        claims=claims,
                        evidence=[],  # can be lazily loaded if needed
                        created_at=r["created_at"],
                        runtime_seconds=r["runtime_seconds"],
                    )
                )
            
            return Session(
                session_id=s_row["session_id"],
                title=s_row["title"],
                created_at=s_row["created_at"],
                updated_at=s_row["updated_at"],
                turns=turns,
            )

    def list_sessions(self) -> List[Session]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
            return [
                Session(
                    session_id=r["session_id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
