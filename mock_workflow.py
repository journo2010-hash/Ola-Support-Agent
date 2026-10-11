"""Mock support workflow for the Ola Domain Support Agent capstone."""

import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from guardrails import check_input, enforce_output_grounding, mask_phone_numbers
from governance import classify_request
from session_memory import SessionMemory
from ticket_lookup import check_support_ticket_status


MODE = "MOCK_LLM"
MAX_INPUT_TOKENS = 400
# Illustrative rate for exercising budget enforcement in MOCK_LLM only.
# It is not a provider price; MOCK_LLM makes no billable model calls.
MOCK_COST_PER_1K_TOKENS_USD = 0.001
MAX_REQUEST_COST_USD = 0.0003
CACHE_TTL_SECONDS = 60

session_memory = SessionMemory(max_turns=10)

WORKFLOW_METRICS: dict[str, float | int] = {
    "workflow_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "policy_tool_calls": 0,
    "ticket_tool_calls": 0,
    "rejected_oversized_requests": 0,
    "rejected_budget_requests": 0,
    "estimated_input_tokens_total": 0,
    "estimated_mock_cost_usd_total": 0.0,
}


def get_workflow_metrics() -> dict[str, float | int]:
    """Return a snapshot of cache, tool-call, and budget counters."""
    return dict(WORKFLOW_METRICS)


def reset_workflow_metrics() -> None:
    """Reset counters for a reproducible demonstration."""
    for key in WORKFLOW_METRICS:
        WORKFLOW_METRICS[key] = 0.0 if "cost" in key else 0


def estimate_mock_cost_usd(token_count: int) -> float:
    """Estimate an illustrative cost unit; actual MOCK_LLM provider cost is $0."""
    return max(0, token_count) / 1000 * MOCK_COST_PER_1K_TOKENS_USD

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
    governance: Optional[dict[str, Any]] = None


def estimate_tokens(text: str) -> int:
    """Conservative, approximate token estimate for input-size control."""
    return max(1, (len(text) + 3) // 4)


def _cache_key(question: str, session_id: str | None = None) -> str:
    """Key the in-memory cache by normalized query text, independent of session."""
    del session_id  # Retained as an optional compatibility argument.
    return re.sub(r"\s+", " ", question.strip().lower())


def _get_cached_response(key: str) -> Optional[dict[str, Any]]:
    """Return a fresh cached response, if one exists."""
    entry = response_cache.get(key)

    if entry is None:
        WORKFLOW_METRICS["cache_misses"] += 1
        return None

    created_at, response = entry

    if time.monotonic() - created_at > CACHE_TTL_SECONDS:
        response_cache.pop(key, None)
        WORKFLOW_METRICS["cache_misses"] += 1
        return None

    WORKFLOW_METRICS["cache_hits"] += 1

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

    # Query cache entries are independent of conversation history.
    return {
        "session_id": session_id,
        "reset": True,
        "cache_entries_removed": 0,
    }


def _make_response(
    question: str,
    answer: str,
    *,
    fallback: bool = False,
    sources: Optional[list[str]] = None,
    tools_used: Optional[list[str]] = None,
    ticket: Optional[dict[str, Any]] = None,
    governance: Optional[dict[str, Any]] = None,
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
        governance=governance,
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
    WORKFLOW_METRICS["policy_tool_calls"] += 1
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

    guarded = enforce_output_grounding(
        answer=answer.strip(),
        sources=sources,
        fallback=fallback,
    )
    return {
        "answer": guarded["answer"],
        "sources": guarded["sources"],
        "fallback": guarded["fallback"],
        "grounded": guarded["grounded"],
        "grounding_guard_reason": guarded["reason"],
    }


def _resolve_session_context(question: str, session_id: str) -> str:
    """Resolve a recent ticket pronoun only when session history is unambiguous."""
    lowered = question.lower()
    follow_up = bool(
        re.search(r"\b(it|its|that ticket|this ticket|the ticket)\b", lowered)
        and re.search(r"\b(status|score|resolution|category|escalat|ticket)\b", lowered)
    )
    if not follow_up:
        return question

    for turn in reversed(session_memory.get_history(session_id)):
        prior_user = str(turn.get("user", ""))
        match = re.search(r"\bOLA-\d{4,}\b", prior_user, re.IGNORECASE)
        if match:
            return f"{question} (context ticket ID: {match.group(0).upper()})"
    return question


def run_support_workflow(
    question: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Run the guarded, cached, mock support-agent workflow."""
    WORKFLOW_METRICS["workflow_calls"] += 1
    original_question = question.strip()
    session_id = (session_id or "default").strip()

    if not original_question:
        return _make_response(
            question="",
            answer="Please enter a support question.",
            fallback=True,
            governance=classify_request(" ").to_dict(),
        )

    # Enforce both an approximate input-token limit and a mock-only budget.
    input_tokens = estimate_tokens(original_question)
    estimated_cost = estimate_mock_cost_usd(input_tokens)
    WORKFLOW_METRICS["estimated_input_tokens_total"] += input_tokens
    WORKFLOW_METRICS["estimated_mock_cost_usd_total"] += estimated_cost

    if input_tokens > MAX_INPUT_TOKENS:
        WORKFLOW_METRICS["rejected_oversized_requests"] += 1
        return _make_response(
            question=mask_phone_numbers(original_question)[:500],
            answer=(
                "Your request exceeds the supported input-token limit. "
                "Please shorten it and try again."
            ),
            fallback=True,
            governance={
                "risk_level": "high",
                "autonomy_level": "blocked",
                "allowed": False,
                "requires_human_review": False,
                "reason": "The request exceeded the configured input-token limit.",
                "max_input_tokens": MAX_INPUT_TOKENS,
                "estimated_input_tokens": input_tokens,
                "estimated_mock_cost_usd": round(estimated_cost, 6),
                "max_request_cost_usd": MAX_REQUEST_COST_USD,
            },
        )

    if estimated_cost > MAX_REQUEST_COST_USD:
        WORKFLOW_METRICS["rejected_budget_requests"] += 1
        return _make_response(
            question=mask_phone_numbers(original_question)[:500],
            answer=(
                "Your request exceeds the configured per-request budget. "
                "Please shorten it and try again."
            ),
            fallback=True,
            governance={
                "risk_level": "high",
                "autonomy_level": "blocked",
                "allowed": False,
                "requires_human_review": False,
                "reason": "The illustrative per-request cost budget was exceeded.",
                "estimated_input_tokens": input_tokens,
                "estimated_mock_cost_usd": round(estimated_cost, 6),
                "max_request_cost_usd": MAX_REQUEST_COST_USD,
            },
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
            governance=classify_request(original_question, guardrail_blocked=True).to_dict(),
        )

    if blocked:
        return _make_response(
            question=safe_question,
            answer=(
                "I can't process this request because it conflicts "
                "with the support agent's safety rules."
            ),
            fallback=True,
            governance=classify_request(safe_question, guardrail_blocked=True).to_dict(),
        )

    # Resolve only a recent ticket reference from this session, when present.
    contextual_question = _resolve_session_context(safe_question, session_id)

    # Classify the request before cache lookup or any tool execution.
    governance_decision = classify_request(contextual_question)
    governance = governance_decision.to_dict()

    if not governance_decision.allowed:
        response = _make_response(
            question=safe_question,
            answer=(
                "This request requires human review and cannot be executed "
                "autonomously by the support agent."
            ),
            fallback=True,
            governance=governance,
        )
        return response

    # Never cache unsafe or oversized inputs. Cache only after guardrails.
    key = _cache_key(contextual_question)
    cached = _get_cached_response(key)

    if cached is not None:
        cached["question"] = safe_question
        session_memory.add_turn(session_id, safe_question, cached["answer"])
        return cached

    # Explicit ticket IDs must always go through the ticket lookup tool.
    if is_ticket_lookup_question(contextual_question):
        ticket_match = re.search(
            r"\bOLA-\d{4,}\b",
            contextual_question,
            re.IGNORECASE,
        )

        if ticket_match is None:
            ticket_result = {"found": False}
        else:
            ticket_id = ticket_match.group(0).upper()
            WORKFLOW_METRICS["ticket_tool_calls"] += 1
            ticket_result = check_support_ticket_status(ticket_id)

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
                governance=governance,
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
            lowered_context = contextual_question.lower()
            if "escalation score" in lowered_context or "escalation risk" in lowered_context:
                answer += f" Escalation score: {ticket['escalation_score']}."
                answer += f" Priority review flag: {ticket['priority_review_flag']}."
            elif "resolution time" in lowered_context:
                answer += f" Resolution time: {ticket['resolution_time_hours']} hours."
            elif "category" in lowered_context:
                answer += f" Category: {ticket['category']}."

            response = _make_response(
                question=safe_question,
                answer=answer,
                fallback=False,
                tools_used=["check_support_ticket_status"],
                ticket=ticket,
                governance=governance,
            )

        # Save the ticket exchange for session history.
        session_memory.add_turn(
            session_id,
            safe_question,
            response["answer"],
        )

        _store_cached_response(key, response)
        return response

    # A ticket pronoun without a ticket ID in this session is unresolved.
    # Do not send this ticket-specific follow-up to policy RAG.
    unresolved_ticket_reference = bool(
        re.search(
            r"\b(it|its|that ticket|this ticket|the ticket)\b",
            safe_question,
            re.IGNORECASE,
        )
        and not is_ticket_lookup_question(contextual_question)
    )
    if unresolved_ticket_reference:
        answer = (
            "I couldn't resolve the ticket reference in this session. "
            "Please provide a ticket ID such as OLA-0001."
        )
        response = _make_response(
            question=safe_question,
            answer=answer,
            fallback=True,
            governance=governance,
        )
        session_memory.add_turn(session_id, safe_question, answer)
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
            governance=governance,
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
        policy_result = search_support_policies(contextual_question)

        response = _make_response(
            question=safe_question,
            answer=(
                policy_result["answer"]
                or "I don't know based on the available support policies."
            ),
            fallback=policy_result["fallback"],
            sources=policy_result["sources"],
            tools_used=["search_support_policies"],
            governance=governance,
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
            governance=governance,
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
        print(f"Governance: {result.get('governance')}")