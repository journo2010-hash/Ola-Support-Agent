
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

# Illustrative local budget rate only. MOCK_LLM makes no billable API calls.
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

# Cache entries: normalized query -> (creation time, response).
response_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# RAG components are initialized lazily and reused.
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


def get_workflow_metrics() -> dict[str, float | int]:
    """Return a snapshot of cache, tool-call, and budget counters."""
    return dict(WORKFLOW_METRICS)


def reset_workflow_metrics() -> None:
    """Reset counters for a reproducible demonstration."""
    for key in WORKFLOW_METRICS:
        WORKFLOW_METRICS[key] = 0.0 if "cost" in key else 0


def estimate_tokens(text: str) -> int:
    """Estimate input tokens for local request-size control."""
    return max(1, (len(text) + 3) // 4)


def estimate_mock_cost_usd(token_count: int) -> float:
    """Estimate illustrative budget units; actual MOCK_LLM cost is zero."""
    return max(0, token_count) / 1000 * MOCK_COST_PER_1K_TOKENS_USD


def _cache_key(question: str, session_id: str | None = None) -> str:
    """Normalize query text for the in-memory cache."""
    del session_id  # Cache remains query-keyed for compatibility.
    return re.sub(r"\s+", " ", question.strip().lower())


def _get_cached_response(key: str) -> Optional[dict[str, Any]]:
    """Return a fresh cached response, if available."""
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

    # Return a copy so callers cannot modify the cached response.
    result = dict(response)
    result["sources"] = list(response.get("sources", []))
    result["tools_used"] = list(response.get("tools_used", []))
    result["cache_hit"] = True
    return result


def _store_cached_response(
    key: str,
    response: dict[str, Any],
) -> None:
    """Store a response without its per-call cache-hit indicator."""
    stored = dict(response)
    stored.pop("cache_hit", None)
    response_cache[key] = (time.monotonic(), stored)


def reset_session(session_id: str = "default") -> dict[str, Any]:
    """Clear one session's history without clearing the query cache."""
    session_memory.clear_session(session_id)
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
    """Validate and serialize a consistent workflow response."""
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
    """Apply input guardrails and return the masked text and block status."""
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

    is_blocked = (
        allowed is False
        or blocked is True
        or injection is True
    )

    return str(masked), is_blocked


def is_ticket_lookup_question(question: str) -> bool:
    """Recognize an explicit support ticket ID."""
    return bool(re.search(r"\bOLA-\d{4,}\b", question, re.IGNORECASE))


def _is_supported_policy_question(question: str) -> bool:
    """Recognize likely support-policy questions."""
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
    """Initialize and cache the local RAG components."""
    global _rag_components

    if _rag_components is None:
        from rag import initialize_support_rag

        components = initialize_support_rag()

        if isinstance(components, dict):
            _rag_components = (
                components["collection"],
                components["model"],
                components["threshold"],
            )
        else:
            _rag_components = tuple(components)

    return _rag_components


def search_support_policies(question: str) -> dict[str, Any]:
    """Search the knowledge base and return grounded policy evidence."""
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

    if isinstance(sources, str):
        sources = [sources]
    else:
        sources = [str(source) for source in sources]

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


def _contains_ticket_follow_up_intent(question: str) -> bool:
    """
    Detect a ticket-specific follow-up without confusing policy questions
    such as 'What is the ticket priority policy?' with ticket references.
    """
    has_reference = bool(
        re.search(
            r"\b(it|its|that ticket|this ticket|the ticket)\b",
            question,
            re.IGNORECASE,
        )
    )

    has_ticket_detail = bool(
        re.search(
            r"\b(status|score|resolution|category|escalat(?:ion|ed|e)?)\b",
            question,
            re.IGNORECASE,
        )
    )

    return has_reference and has_ticket_detail


def _resolve_session_context(question: str, session_id: str) -> str:
    """Resolve a ticket follow-up only when session history is unambiguous."""
    if not _contains_ticket_follow_up_intent(question):
        return question

    for turn in reversed(session_memory.get_history(session_id)):
        prior_user = str(turn.get("user", ""))
        match = re.search(
            r"\bOLA-\d{4,}\b",
            prior_user,
            re.IGNORECASE,
        )
        if match:
            return (
                f"{question} "
                f"(context ticket ID: {match.group(0).upper()})"
            )

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

    # Enforce the input-size and illustrative mock-budget limits.
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
                "reason": (
                    "The illustrative per-request cost budget was exceeded."
                ),
                "estimated_input_tokens": input_tokens,
                "estimated_mock_cost_usd": round(estimated_cost, 6),
                "max_request_cost_usd": MAX_REQUEST_COST_USD,
            },
        )

    # Apply phone masking and prompt-injection checks.
    try:
        safe_question, blocked = _guard_question(original_question)
    except Exception:
        return _make_response(
            question="[INPUT REDACTED]",
            answer=(
                "I can't process this request safely right now. "
                "Please try again."
            ),
            fallback=True,
            governance=classify_request(
                original_question,
                guardrail_blocked=True,
            ).to_dict(),
        )

    if blocked:
        return _make_response(
            question=safe_question,
            answer=(
                "I can't process this request because it conflicts "
                "with the support agent's safety rules."
            ),
            fallback=True,
            governance=classify_request(
                safe_question,
                guardrail_blocked=True,
            ).to_dict(),
        )

    # Resolve context only for a genuine ticket-specific follow-up.
    contextual_question = _resolve_session_context(
        safe_question,
        session_id,
    )

    governance_decision = classify_request(contextual_question)
    governance = governance_decision.to_dict()

    if not governance_decision.allowed:
        return _make_response(
            question=safe_question,
            answer=(
                "This request requires human review and cannot be "
                "executed autonomously by the support agent."
            ),
            fallback=True,
            governance=governance,
        )

    # Cache only after the input guardrails and governance checks.
    key = _cache_key(contextual_question)
    cached = _get_cached_response(key)

    if cached is not None:
        cached["question"] = safe_question
        session_memory.add_turn(
            session_id,
            safe_question,
            cached["answer"],
        )
        return cached

    # An explicit ticket ID takes the ticket-lookup path.
    if is_ticket_lookup_question(contextual_question):
        ticket_match = re.search(
            r"\bOLA-\d{4,}\b",
            contextual_question,
            re.IGNORECASE,
        )

        ticket_result = {"found": False}

        if ticket_match is not None:
            ticket_id = ticket_match.group(0).upper()
            WORKFLOW_METRICS["ticket_tool_calls"] += 1
            ticket_result = check_support_ticket_status(ticket_id)

        if not ticket_result.get("found", False):
            response = _make_response(
                question=safe_question,
                answer="Ticket not found. Please check the ticket ID.",
                fallback=True,
                tools_used=["check_support_ticket_status"],
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

            lowered = contextual_question.lower()

            if (
                "escalation score" in lowered
                or "escalation risk" in lowered
                or "priority review" in lowered
            ):
                answer += f" Escalation score: {ticket['escalation_score']}."
                answer += (
                    " Priority review flag: "
                    f"{ticket['priority_review_flag']}."
                )
            elif "resolution time" in lowered:
                answer += (
                    f" Resolution time: "
                    f"{ticket['resolution_time_hours']} hours."
                )
            elif "category" in lowered:
                answer += f" Category: {ticket['category']}."

            response = _make_response(
                question=safe_question,
                answer=answer,
                fallback=False,
                tools_used=["check_support_ticket_status"],
                ticket=ticket,
                governance=governance,
            )

        session_memory.add_turn(
            session_id,
            safe_question,
            response["answer"],
        )

        _store_cached_response(key, response)
        return response

    # A ticket-specific follow-up without a resolvable ID needs clarification.
    # General policy questions such as 'ticket priority policy' do not.
    if (
        _contains_ticket_follow_up_intent(safe_question)
        and not is_ticket_lookup_question(contextual_question)
    ):
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

    # Reject unrelated requests instead of inventing support policies.
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

    # Supported policy questions use grounded RAG retrieval.
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
