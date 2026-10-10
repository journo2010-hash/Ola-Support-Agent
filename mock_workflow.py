
"""Mock support workflow for the Ola Domain Support Agent capstone."""

import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from guardrails import check_input
from session_memory import SessionMemory
from ticket_lookup import check_support_ticket_status


MODE = "MOCK_LLM"
MAX_INPUT_TOKENS = 400
CACHE_TTL_SECONDS = 60

session_memory = SessionMemory(max_turns=10)

# Cache entries: key -> (created_at, response)
response_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# RAG components are initialized only when needed.
_rag_components: Optional[tuple] = None


class WorkflowResponse(BaseModel):
    """Validated response returned by the support workflow."""

    answer: str
    mode: str = MODE
    fallback: bool = False
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    ticket: Optional[dict[str, Any]] = None
    question: str


def estimate_tokens(text: str) -> int:
    """Conservative, approximate token estimate for input-size control."""
    return max(1, (len(text) + 3) // 4)


def _cache_key(question: str, session_id: str) -> str:
    """Normalize the question before building its session-specific cache key."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return f"{session_id.strip()}::{normalized}"


def _get_cached_response(key: str) -> Optional[dict[str, Any]]:
    """Return a fresh cached response, if one exists."""
    entry = response_cache.get(key)

    if entry is None:
        return None

    created_at, response = entry

    if time.monotonic() - created_at > CACHE_TTL_SECONDS:
        response_cache.pop(key, None)
        return None

    # Return a copy so callers cannot modify the cached object.
    result = dict(response)
    result["sources"] = list(response.get("sources", []))
    result["tools_used"] = list(response.get("tools_used", []))
    result["cache_hit"] = True
    return result


def _store_cached_response(
    key: str,
    response: dict[str, Any],
) -> None:
    """Cache a successful workflow result."""
    stored = dict(response)
    stored.pop("cache_hit", None)
    response_cache[key] = (time.monotonic(), stored)


def reset_session(session_id: str = "default") -> dict[str, Any]:
    """Clear one session's conversation history and cached responses."""
    session_memory.clear_session(session_id)

    prefix = f"{session_id.strip()}::"
    keys_to_remove = [
        key for key in response_cache if key.startswith(prefix)
    ]

    for key in keys_to_remove:
        response_cache.pop(key, None)

    return {
        "session_id": session_id,
        "reset": True,
        "cache_entries_removed": len(keys_to_remove),
    }


def _make_response(
    question: str,
    answer: str,
    *,
    fallback: bool = False,
    sources: Optional[list[str]] = None,
    tools_used: Optional[list[str]] = None,
    ticket: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Validate and serialize a consistent API response."""
    result = WorkflowResponse(
        answer=answer,
        mode=MODE,
        fallback=fallback,
        sources=sources or [],
        tools_used=tools_used or [],
        ticket=ticket,
        question=question,
    )
    return result.model_dump()


def _guard_question(question: str) -> tuple[str, bool]:
    """
    Apply input guardrails.

    Supports a dictionary or an object returned by check_input().
    Returns the masked question and whether processing is blocked.
    """
    result = check_input(question)

    if isinstance(result, dict):
        masked = result.get("masked_text", question)
        allowed = result.get("allowed", True)
        blocked = result.get("blocked", False)
        injection = result.get("is_injection", False)
    else:
        masked = getattr(result, "masked_text", question)
        allowed = getattr(result, "allowed", True)
        blocked = getattr(result, "blocked", False)
        injection = getattr(result, "is_injection", False)

    # Do not continue if the guardrail explicitly blocks the request.
    is_blocked = (
        allowed is False
        or blocked is True
        or injection is True
    )

    return str(masked), is_blocked


def is_ticket_lookup_question(question: str) -> bool:
    """Recognize an explicit Ola support ticket ID."""
    return bool(re.search(r"\bOLA-\d{4,}\b", question, re.IGNORECASE))


def _is_supported_policy_question(question: str) -> bool:
    """Route likely support-policy questions to the RAG workflow."""
    patterns = [
        r"\bsupport\b",
        r"\bticket\b",
        r"\bpriorit(?:y|ies|ize|isation|ization)\b",
        r"\bP[1-4]\b",
        r"\bSLA\b",
        r"\bservice.level\b",
        r"\bescalat",
        r"\brefund",
        r"\bcompensation\b",
        r"\bcommunication\b",
        r"\bchannels?\b",
        r"\bbusiness.hours\b",
        r"\bholidays?\b",
        r"\brepeat.complaint",
        r"\bservice.credits?\b",
        r"\bfeedback\b",
        r"\bVIP\b",
        r"\boutage\b",
        r"\bretention\b",
        r"\bdata.policy\b",
        r"\bcomplaint",
        r"\bresolution.time\b",
    ]

    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


def get_rag_components() -> tuple:
    """Initialize the RAG components once and reuse them."""
    global _rag_components

    if _rag_components is None:
        from rag import initialize_support_rag

        components = initialize_support_rag()

        if isinstance(components, dict):
            collection = components["collection"]
            model = components["model"]
            threshold = components["threshold"]
            _rag_components = (collection, model, threshold)
        else:
            _rag_components = tuple(components)

    return _rag_components


def search_support_policies(question: str) -> dict[str, Any]:
    """Search the support knowledge base and return grounded results."""
    from rag import grounded_answer

    components = get_rag_components()

    if len(components) < 3:
        raise RuntimeError(
            "RAG initialization must provide collection, model, and threshold."
        )

    collection, model, threshold = components[:3]

    result = grounded_answer(
        question,
        collection,
        model,
        threshold,
    )

    answer = ""
    sources: list[str] = []
    fallback = False

    if isinstance(result, dict):
        answer = str(result.get("answer", ""))
        sources = result.get("sources", []) or []
        fallback = bool(result.get("fallback", False))

    elif isinstance(result, tuple):
        if len(result) >= 1:
            answer = str(result[0])
        if len(result) >= 2:
            sources = result[1] or []
        if len(result) >= 3:
            fallback = bool(result[2])

    else:
        answer = str(result)

    # Normalize source names to a list of strings.
    if isinstance(sources, str):
        sources = [sources]
    else:
        sources = [str(source) for source in sources]

    # Recognize the project's established RAG refusal wording.
    refusal_phrases = (
        "i don't know based on the available support policies",
        "i do not know based on the available support policies",
        "couldn't identify a relevant policy",
        "could not identify a relevant policy",
    )

    if not answer.strip() or any(
        phrase in answer.lower() for phrase in refusal_phrases
    ):
        fallback = True

    return {
        "answer": answer.strip(),
        "sources": sources,
        "fallback": fallback,
    }


def run_support_workflow(
    question: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Run the guarded, cached, mock support-agent workflow."""
    original_question = question.strip()
    session_id = (session_id or "default").strip()

    if not original_question:
        return _make_response(
            question="",
            answer="Please enter a support question.",
            fallback=True,
        )

    # Reject oversized requests before invoking downstream components.
    if estimate_tokens(original_question) > MAX_INPUT_TOKENS:
        return _make_response(
            question=original_question[:500],
            answer=(
                "Your request exceeds the supported input limit. "
                "Please shorten it and try again."
            ),
            fallback=True,
        )

    # Guardrails mask phone numbers and detect prompt injection.
    try:
        safe_question, blocked = _guard_question(original_question)
    except Exception:
        # Fail closed if guardrail evaluation itself fails.
        return _make_response(
            question="[INPUT REDACTED]",
            answer=(
                "I can't process this request safely right now. "
                "Please try again."
            ),
            fallback=True,
        )

    if blocked:
        return _make_response(
            question=safe_question,
            answer=(
                "I can't process this request because it conflicts "
                "with the support agent's safety rules."
            ),
            fallback=True,
        )

    # Never cache unsafe or oversized inputs. Cache only after guardrails.
    key = _cache_key(safe_question, session_id)
    cached = _get_cached_response(key)

    if cached is not None:
        return cached

    # Explicit ticket IDs must always go through the ticket lookup tool.
    if is_ticket_lookup_question(safe_question):
        ticket_result = check_support_ticket_status(safe_question)

        if not ticket_result.get("found", False):
            # FIX: A missing ticket is a fallback, not a successful lookup.
            response = _make_response(
                question=safe_question,
                answer=(
                    "Ticket not found. Please check the ticket ID."
                ),
                fallback=True,
                tools_used=["check_support_ticket_status"],
                ticket=None,
            )
        else:
            ticket = {
                key: value
                for key, value in ticket_result.items()
                if key != "found"
            }

            answer = (
                f"Ticket {ticket['record_id']} is currently "
                f"{ticket['status']}."
            )

            response = _make_response(
                question=safe_question,
                answer=answer,
                fallback=False,
                tools_used=["check_support_ticket_status"],
                ticket=ticket,
            )

        # Save the ticket exchange for session history.
        session_memory.add_turn(
            session_id,
            safe_question,
            response["answer"],
        )

        _store_cached_response(key, response)
        return response

    # Avoid inventing answers for unrelated questions.
    if not _is_supported_policy_question(safe_question):
        answer = (
            "I couldn't identify a ticket or supported policy topic. "
            "Please provide a ticket ID or ask about a support policy."
        )

        response = _make_response(
            question=safe_question,
            answer=answer,
            fallback=True,
        )

        session_memory.add_turn(
            session_id,
            safe_question,
            answer,
        )

        _store_cached_response(key, response)
        return response

    # Search the knowledge base for supported policy questions.
    try:
        policy_result = search_support_policies(safe_question)

        response = _make_response(
            question=safe_question,
            answer=(
                policy_result["answer"]
                or "I don't know based on the available support policies."
            ),
            fallback=policy_result["fallback"],
            sources=policy_result["sources"],
            tools_used=["search_support_policies"],
        )

    except Exception:
        # Do not expose internal errors or invent policy information.
        response = _make_response(
            question=safe_question,
            answer=(
                "I couldn't retrieve the support policy information "
                "right now. Please try again."
            ),
            fallback=True,
            tools_used=["search_support_policies"],
        )

    session_memory.add_turn(
        session_id,
        safe_question,
        response["answer"],
    )

    _store_cached_response(key, response)
    return response


if __name__ == "__main__":
    print("Ola Support Agent — MOCK_LLM mode")
    print("Type 'reset' to clear the current session, or 'exit' to quit.")

    current_session = "cli-session"

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if question.lower() == "reset":
            result = reset_session(current_session)
            print(f"Session reset: {result}")
            continue

        result = run_support_workflow(
            question,
            session_id=current_session,
        )

        print(f"\nAgent: {result['answer']}")
        print(f"Fallback: {result['fallback']}")
        print(f"Sources: {result['sources']}")
        print(f"Tools used: {result['tools_used']}")
