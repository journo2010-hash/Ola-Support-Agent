"""Manage multi-turn support conversations using LangChain chat history."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class SessionMemory:
    """Maintain separate, bounded conversation histories for each session."""

    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.sessions = {}

    def _get_session(self, session_id):
        """Get or create a session's LangChain chat history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]

    def add_turn(self, session_id, user_message, assistant_message):
        """Store a complete user and assistant exchange."""
        history = self._get_session(session_id)

        history.add_messages([
            HumanMessage(content=user_message),
            AIMessage(content=assistant_message),
        ])

        max_messages = self.max_turns * 2

        if len(history.messages) > max_messages:
            history.messages = history.messages[-max_messages:]

    def get_history(self, session_id):
        """Return conversation turns in the format expected by mock_workflow.py."""
        if session_id not in self.sessions:
            return []

        messages = self.sessions[session_id].messages
        turns = []

        for index in range(0, len(messages) - 1, 2):
            user_message = messages[index]
            assistant_message = messages[index + 1]

            if (
                isinstance(user_message, HumanMessage)
                and isinstance(assistant_message, AIMessage)
            ):
                turns.append({
                    "user": user_message.content,
                    "assistant": assistant_message.content,
                })

        return turns

    def clear_session(self, session_id):
        """Clear one session without affecting other sessions."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()


if __name__ == "__main__":
    memory = SessionMemory(max_turns=2)

    memory.add_turn(
        "demo",
        "What is the status of OLA-0001?",
        "The ticket is Open.",
    )
    memory.add_turn(
        "demo",
        "What is its escalation score?",
        "The escalation score is 0.173.",
    )

    history = memory.get_history("demo")

    assert len(history) == 2
    assert history[0]["user"] == "What is the status of OLA-0001?"
    assert history[1]["assistant"] == "The escalation score is 0.173."
    assert memory.get_history("other-session") == []

    memory.clear_session("demo")
    assert memory.get_history("demo") == []

    print("LANGCHAIN SESSION MEMORY TEST: PASSED")
    print("Conversation history format: PASSED")
    print("Session isolation: PASSED")
    print("Session reset: PASSED")
