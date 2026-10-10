"""Simple in-memory conversation history for support sessions."""

from collections import defaultdict
from threading import RLock


class SessionMemory:
    """Store conversation turns separately for each session."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._sessions = defaultdict(list)
        self._lock = RLock()

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Save one user/assistant conversation turn."""

        with self._lock:
            history = self._sessions[session_id]
            history.append({
                "user": user_message,
                "assistant": assistant_response,
            })
            self._sessions[session_id] = history[-self.max_turns:]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of the session's conversation history."""

        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Reset one session without affecting other sessions."""

        with self._lock:
            self._sessions.pop(session_id, None)


if __name__ == "__main__":
    memory = SessionMemory(max_turns=3)

    memory.add_turn(
        "session-1",
        "What is the status of OLA-0001?",
        "Ticket OLA-0001 is Escalated.",
    )
    memory.add_turn(
        "session-1",
        "What is its escalation score?",
        "The escalation score is 0.173.",
    )

    print("Session memory test:")
    print(memory.get_history("session-1"))

    memory.clear_session("session-1")
    print("After reset:", memory.get_history("session-1"))

    assert memory.get_history("session-1") == []
    print("SESSION MEMORY TEST: SUCCESS")