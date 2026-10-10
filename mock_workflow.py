
"""Deterministic local workflow using the real support tools."""

import json
import re

from crew_support import search_support_policies, ticket_status_tool


def run_support_workflow(question: str) -> dict:
    """Call local tools explicitly and return their verified results."""

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a support question.",
            "tools_used": [],
        }

    result = {
        "question": question,
        "mode": "MOCK_LLM",
        "tools_used": [],
    }

    # Find a ticket ID, if one is present.
    match = re.search(r"\bOLA-\d{4}\b", question, re.IGNORECASE)

    if match:
        ticket_id = match.group(0).upper()
        ticket_result = json.loads(
            ticket_status_tool.run(record_id=ticket_id)
        )
        result["ticket_lookup"] = ticket_result
        result["tools_used"].append("check_support_ticket_status")

        if not ticket_result.get("found"):
            result["answer"] = ticket_result["message"]
            return result

    # Search policies only when the question asks about a policy.
    policy_keywords = (
        "policy", "priorit", "severity", "sla", "response time",
        "escalat", "refund", "business hours", "outage",
        "communication", "retention", "vip", "service credit",
        "feedback", "complaint", "support channel",
    )
    asks_policy = any(word in question.lower() for word in policy_keywords)

    if asks_policy or not match:
        policy_result = json.loads(
            search_support_policies.run(question=question)
        )
        result["policy_result"] = policy_result
        result["tools_used"].append("search_support_policies")

        if policy_result.get("fallback"):
            result["answer"] = (
                "I don't know based on the available support policies."
            )
            return result

    if match:
        ticket = result["ticket_lookup"]
        result["answer"] = (
            f"Ticket {ticket['record_id']} is currently "
            f"{ticket['status']}. Category: {ticket['category']}. "
            f"Resolution time: {ticket['resolution_time_hours']} hours. "
            f"Escalation score: {ticket['escalation_score']}."
        )
    else:
        result["answer"] = result["policy_result"]["answer"]

    return result


if __name__ == "__main__":
    question = input("Enter a support question or ticket ID: ")
    print(json.dumps(run_support_workflow(question), indent=2))
