"""Ola support workflow with memory, guardrails, caching, and input limits."""

import copy
import json
import re
import time
from typing import Any

from crew_support import search_support_policies, ticket_status_tool
from response_models import validate_support_response
from session_memory import SessionMemory
from guardrails import check_input


# Runtime governance limits.
MAX_INPUT_TOKENS = 400
CACHE_TTL_SECONDS = 60

session_memory = SessionMemory(max_turns=10)

# In-memory cache: (session ID, normalized question) -> response and timestamp.
response_cache = {}


def estimate_tokens(text: str) -> int:
    """Estimate input tokens without an external tokenizer or network call."""
    return (len(text) + 3) // 4


def normalize_question(question: str) -> str:
    """Normalize whitespace and capitalization for cache matching."""
    return " ".join(question.lower().split())


def is_cacheable(question: str) -> bool:
    """Cache only self-contained ticket or policy questions."""

    if re.search(r"\bOLA-\d{4}\b", question, re.IGNORECASE):
        return True

    follow_up_phrases = (
        "it",
        "its",
        "that ticket",
        "same ticket",
        "escalation score",
        "resolution time",
        "ticket status",
        "status of it",
    )

    lower_question = question.lower()

    # Do not cache context-dependent follow-up questions.
    if any(phrase in lower_question for phrase in follow_up_phrases):
        return False

    policy_keywords = (
        "policy", "priorit", "severity", "sla", "response time",
        "refund", "business hours", "outage", "communication",
        "retention", "vip", "service credit", "feedback",
        "complaint", "support channel",
    )

    return any(word in lower_question for word in policy_keywords)


def make_response(
    answer: str,
    question: str = "",
    tools_used: list[str] | None = None,
    fallback: bool = False,
    sources: list[str] | None = None,
    ticket: dict[str, Any] | None = None,
) -> dict:
    """Create and validate a structured support response."""

    data = {
        "answer": answer,
        "mode": "MOCK_LLM",
        "fallback": fallback,
        "sources": sources or [],
        "tools_used": tools_used or [],
        "ticket": ticket,
    }

    result = validate_support_response(data).model_dump()
    result["question"] = question
    result["cache_hit"] = False
    return result


def run_support_workflow(
    question: str,
    session_id: str = "default",
) -> dict:
    """Apply guardrails and input limits, then retrieve or reuse a response."""

    question = question.strip()

    if not question:
        return make_response(
            "Please enter a support question.",
            question="",
            fallback=True,
        )

    # Guardrails run before caching or tool invocation.
    guardrail = check_input(question)

    if guardrail.blocked:
        return make_response(
            answer=(
                "I can't process this request because it contains "
                "instructions that conflict with the support agent's rules."
            ),
            question=guardrail.masked_text,
            fallback=True,
        )

    # Use masked text throughout the rest of the workflow.
    question = guardrail.masked_text

    # Reject oversized inputs before calling any support tools.
    estimated_tokens = estimate_tokens(question)

    if estimated_tokens > MAX_INPUT_TOKENS:
        return make_response(
            answer=(
                f"Your request exceeds the input limit of approximately "
                f"{MAX_INPUT_TOKENS} tokens. Please shorten your question."
            ),
            question="[INPUT REJECTED: input limit exceeded]",
            fallback=True,
        )

    normalized = normalize_question(question)
    cache_key = (session_id, normalized)
    now = time.monotonic()

    # Cache only self-contained questions to avoid reusing stale context
    # for follow-up questions that depend on conversation history.
    if is_cacheable(question):
        cached = response_cache.get(cache_key)

        if cached:
            age = now - cached["created_at"]

            if age <= CACHE_TTL_SECONDS:
                response = copy.deepcopy(cached["response"])
                response["cache_hit"] = True

                # No tools ran for this request; the result came from cache.
                response["tools_used"] = []

                session_memory.add_turn(
                    session_id,
                    question,
                    response["answer"],
                )

                return response

            del response_cache[cache_key]

    history = session_memory.get_history(session_id)

    tools_used = []
    sources = []
    ticket_details = None
    policy_result = None
    fallback = False

    match = re.search(r"\bOLA-\d{4}\b", question, re.IGNORECASE)
    ticket_id = match.group(0).upper() if match else None
    remembered_ticket = False

    # Recover a ticket ID for simple follow-up questions.
    if ticket_id is None and history:
        follow_up_phrases = (
            "it",
            "its",
            "that ticket",
            "same ticket",
            "escalation score",
            "resolution time",
            "ticket status",
            "status of it",
        )

        if any(phrase in question.lower() for phrase in follow_up_phrases):
            for turn in reversed(history):
                previous_text = turn["user"] + " " + turn["assistant"]
                previous_match = re.search(
                    r"\bOLA-\d{4}\b",
                    previous_text,
                    re.IGNORECASE,
                )

                if previous_match:
                    ticket_id = previous_match.group(0).upper()
                    remembered_ticket = True
                    break

    # Ticket lookup.
    if ticket_id:
        ticket_result = json.loads(
            ticket_status_tool.run(record_id=ticket_id)
        )
        tools_used.append("check_support_ticket_status")

        if not ticket_result.get("found"):
            response = make_response(
                answer=ticket_result.get(
                    "message",
                    f"Ticket {ticket_id} was not found.",
                ),
                question=question,
                tools_used=tools_used,
            )

            session_memory.add_turn(
                session_id,
                question,
                response["answer"],
            )

            return response

        ticket_details = {
            "record_id": ticket_result["record_id"],
            "status": ticket_result["status"],
            "category": ticket_result["category"],
            "resolution_time_hours": ticket_result[
                "resolution_time_hours"
            ],
            "escalation_score": ticket_result["escalation_score"],
        }

    policy_keywords = (
        "policy", "priorit", "severity", "sla", "response time",
        "refund", "business hours", "outage", "communication",
        "retention", "vip", "service credit", "feedback",
        "complaint", "support channel",
    )

    asks_policy = any(
        word in question.lower() for word in policy_keywords
    )

    # Avoid unnecessary RAG calls for ticket follow-up questions.
    if asks_policy and not remembered_ticket:
        policy_result = json.loads(
            search_support_policies.run(question=question)
        )
        tools_used.append("search_support_policies")
        sources = policy_result.get("sources", [])
        fallback = bool(policy_result.get("fallback", False))

        if fallback:
            response = make_response(
                answer=(
                    "I don't know based on the available support policies."
                ),
                question=question,
                tools_used=tools_used,
                fallback=True,
            )

            session_memory.add_turn(
                session_id,
                question,
                response["answer"],
            )

            return response

    if ticket_details:
        lower_question = question.lower()

        if "escalation score" in lower_question:
            answer = (
                f"Ticket {ticket_id} has an escalation score of "
                f"{ticket_details['escalation_score']}."
            )

        elif "resolution time" in lower_question:
            answer = (
                f"Ticket {ticket_id} has a recorded resolution time of "
                f"{ticket_details['resolution_time_hours']} hours."
            )

        elif "status" in lower_question or "state" in lower_question:
            answer = (
                f"Ticket {ticket_id} is currently "
                f"{ticket_details['status']}."
            )

        else:
            answer = (
                f"Ticket {ticket_id} is currently "
                f"{ticket_details['status']}. "
                f"Category: {ticket_details['category']}. "
                f"Resolution time: "
                f"{ticket_details['resolution_time_hours']} hours. "
                f"Escalation score: "
                f"{ticket_details['escalation_score']}."
            )

    elif policy_result:
        answer = policy_result["answer"]

    else:
        answer = (
            "I couldn't identify a ticket or supported policy topic. "
            "Please provide a ticket ID or ask about a support policy."
        )
        fallback = True

    response = make_response(
        answer=answer,
        question=question,
        tools_used=tools_used,
        fallback=fallback,
        sources=sources,
        ticket=ticket_details,
    )

    session_memory.add_turn(
        session_id,
        question,
        response["answer"],
    )

    # Cache eligible responses for a short, bounded period.
    if is_cacheable(question):
        response_cache[cache_key] = {
            "created_at": time.monotonic(),
            "response": copy.deepcopy(response),
        }

    return response


if __name__ == "__main__":
    current_session = "demo-session"

    while True:
        question = input(
            "\nEnter a support question (or type 'reset' or 'exit'): "
        )

        if question.strip().lower() == "exit":
            break

        if question.strip().lower() == "reset":
            session_memory.clear_session(current_session)

            # Reset this session's cached responses too.
            keys_to_remove = [
                key for key in response_cache
                if key[0] == current_session
            ]

            for key in keys_to_remove:
                del response_cache[key]

            print("Session memory and cache cleared.")
            continue

        result = run_support_workflow(
            question,
            session_id=current_session,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))