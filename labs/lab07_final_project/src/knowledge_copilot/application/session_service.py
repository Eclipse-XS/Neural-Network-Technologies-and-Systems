"""Session management application service."""
import uuid
from typing import List, Optional

from ..domain.models import Session
from ..infrastructure.sqlite_repo import SQLiteRepository


class SessionService:
    def __init__(self, sqlite_repo: SQLiteRepository):
        self.sqlite_repo = sqlite_repo

    def create_session(self, session_id: Optional[str] = None, title: str = "New Conversation") -> Session:
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        return self.sqlite_repo.create_session(sid, title=title)

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sqlite_repo.get_session(session_id)

    def list_sessions(self) -> List[Session]:
        return self.sqlite_repo.list_sessions()
