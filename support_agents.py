
"""Ola Support Agent: simple local mock workflow."""

import os

os.environ["MOCK_LLM"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


def run_mock_support_workflow(question: str) -> dict:
    """Answer a policy question or look up a ticket without an API key."""

    from rag import grounded_answer, initialize_support_rag
    from ticket_lookup import check_support_ticket_status

    question = question.strip()

    if not question:
        return {"answer": "Please enter a question.", "tools_used": []}

    # Initialize RAG when the workflow is first called.
    collection, model, threshold = initialize_support_rag()

    result = {
        "question": question,
        "mode": "MOCK_LLM",
        "tools_used": [],
    }

    # Look up a ticket if the question includes an OLA ticket ID.
    ticket_id = next(
        (
            word.strip(".,!?():")
            for word in question.upper().split()
            if word.strip(".,!?():").startswith("OLA-")
        ),
        None,
    )

    if ticket_id:
        result["ticket_lookup"] = check_support_ticket_status(ticket_id)
        result["tools_used"].append("check_support_ticket_status")

    # Retrieve policy information from the knowledge base.
    policy = grounded_answer(question, collection, model, threshold)
    result["policy_result"] = policy
    result["tools_used"].append("grounded_answer")
    result["answer"] = policy["answer"]

    return result


if __name__ == "__main__":
    question = input("Enter a support question or ticket ID: ")
    print(run_mock_support_workflow(question))
