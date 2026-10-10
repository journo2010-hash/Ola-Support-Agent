
"""Ola support workflow with memory, validation, and input guardrails."""

import json
import re
from typing import Any

from crew_support import search_support_policies, ticket_status_tool
from response_models import validate_support_response
from session_memory import SessionMemory
from guardrails import check_input

session_memory = SessionMemory(max_turns=10)


def make_response(
    answer: str,
    question: str = "",
    tools_used: list[str] | None = None,
    fallback: bool = False,
    sources: list[str] | None = None,
    ticket: dict[str, Any] | None = None,
) -> dict:
    """Create a response and validate it with Pydantic."""

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
    return result


def run_support_workflow(
    question: str,
    session_id: str = "default",
) -> dict:
    """Check inputs, then use ticket lookup or policy retrieval."""

    question = question.strip()

    if not question:
        return make_response(
            "Please enter a support question.",
            question=question,
        )

    # Check the original input before sending it to tools.
    guardrail = check_input(question)

    # Block detected injection attempts.
    if guardrail.blocked:
        return make_response(
            answer=(
                "I can't process this request because it contains "
                "instructions that conflict with the support agent's rules."
            ),
            question=guardrail.masked_text,
            fallback=True,
        )

    # Use masked text for all subsequent processing.
    question = guardrail.masked_text
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

        if any(p in question.lower() for p in follow_up_phrases):
            for turn in reversed(history):
                previous_text = turn["user"] + " " + turn["assistant"]
                previous_match = re.search(
                    r"\bOLA-\d{4}\b", previous_text, re.IGNORECASE
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
                    "message", f"Ticket {ticket_id} was not found."
                ),
                question=question,
                tools_used=tools_used,
            )
            session_memory.add_turn(
                session_id, question, response["answer"]
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

    # Don't run RAG for a remembered-ticket follow-up.
    if asks_policy and not remembered_ticket:
        policy_result = json.loads(
            search_support_policies.run(question=question)
        )
        tools_used.append("search_support_policies")
        sources = policy_result.get("sources", [])
        fallback = bool(policy_result.get("fallback", False))

        if fallback:
            response = make_response(
                answer="I don't know based on the available support policies.",
                question=question,
                tools_used=tools_used,
                fallback=True,
            )
            session_memory.add_turn(
                session_id, question, response["answer"]
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
        session_id, question, response["answer"]
    )
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
            print("Session memory cleared.")
            continue

        result = run_support_workflow(
            question, session_id=current_session
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
