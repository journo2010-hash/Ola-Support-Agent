
"""Three-agent CrewAI team with deterministic local tool execution."""

import json
import os
import re

os.environ["MOCK_LLM"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from mock_llm import MockSupportLLM
from response_models import SupportResponse, validate_support_response
from ticket_lookup import check_support_ticket_status


_rag_components = None
TOOL_EXECUTION_LOG = []
TOOL_RESULT_LOG: dict[str, dict] = {}


def get_rag_components():
    """Initialize the local RAG components once."""
    global _rag_components

    if _rag_components is None:
        from rag import initialize_support_rag

        _rag_components = initialize_support_rag()

    return _rag_components


@tool("search_support_policies")
def search_support_policies(question: str) -> str:
    """Search support policy documents and return grounded evidence."""
    from rag import grounded_answer

    collection, model, threshold = get_rag_components()
    result = grounded_answer(question, collection, model, threshold)

    TOOL_EXECUTION_LOG.append("search_support_policies")
    TOOL_RESULT_LOG["policy"] = result
    return json.dumps(result, ensure_ascii=False)


@tool("check_support_ticket_status")
def ticket_status_tool(record_id: str) -> str:
    """Look up a ticket such as OLA-0001."""
    result = check_support_ticket_status(record_id)

    TOOL_EXECUTION_LOG.append("check_support_ticket_status")
    TOOL_RESULT_LOG["ticket"] = result
    return json.dumps(result, ensure_ascii=False)


def build_support_crew():
    """Build the three-agent, three-task sequential crew."""
    retrieval_agent = Agent(
        role="Support Policy Retrieval Specialist",
        goal="Retrieve policy evidence using the policy search tool.",
        backstory=(
            "You rely on retrieved policy evidence and never invent policies."
        ),
        llm=MockSupportLLM(),
        tools=[search_support_policies],
        verbose=False,
        allow_delegation=False,
    )

    lookup_agent = Agent(
        role="Support Ticket Lookup Specialist",
        goal="Retrieve exact ticket information using the ticket tool.",
        backstory="You report only ticket information returned by the tool.",
        llm=MockSupportLLM(),
        tools=[ticket_status_tool],
        verbose=False,
        allow_delegation=False,
    )

    composer_agent = Agent(
        role="Support Response Composer",
        goal="Compose a concise answer from verified task results.",
        backstory=(
            "You must not invent information absent from tool results."
        ),
        llm=MockSupportLLM(),
        verbose=False,
        allow_delegation=False,
    )

    retrieval_task = Task(
        description=(
            "Retrieve policy evidence for this question: {question}. "
            "Use the search_support_policies tool and report its exact result."
        ),
        expected_output=(
            "Policy evidence and source filenames from the retrieval tool."
        ),
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=(
            "Inspect this question: {question}. If it contains a ticket ID, "
            "call check_support_ticket_status with that ID and report the "
            "exact returned result. If no ID is present, say so."
        ),
        expected_output="Exact ticket lookup result, or no ticket ID.",
        agent=lookup_agent,
    )

    composer_task = Task(
        description=(
            "Answer the question: {question}. Use only the preceding task "
            "outputs. Include actual ticket fields when available. For a "
            "ticket lookup, state its status, category, resolution time, "
            "and escalation score. For policy questions, summarize retrieved "
            "evidence and cite source filenames. Do not invent details."
        ),
        expected_output=(
            "A concise answer grounded in the preceding task outputs."
        ),
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )


def validate_crew_response(raw_result: Any) -> SupportResponse:
    """Validate every final CrewAI kickoff result with the Pydantic schema."""
    raw_answer = str(raw_result).strip() or (
        "The support workflow did not return an answer."
    )
    policy_result = TOOL_RESULT_LOG.get("policy", {})
    ticket_result = TOOL_RESULT_LOG.get("ticket", {})

    ticket = None
    if ticket_result.get("found"):
        ticket = {
            "record_id": ticket_result["record_id"],
            "status": ticket_result["status"],
            "category": ticket_result["category"],
            "resolution_time_hours": ticket_result["resolution_time_hours"],
            "escalation_score": ticket_result["escalation_score"],
        }

    sources = policy_result.get("sources", []) or []
    fallback = bool(policy_result.get("fallback", False)) and ticket is None
    return validate_support_response({
        "answer": raw_answer,
        "mode": "MOCK_LLM",
        "fallback": fallback,
        "sources": [str(source) for source in sources],
        "tools_used": list(dict.fromkeys(TOOL_EXECUTION_LOG)),
        "ticket": ticket,
    })


def run_local_tools(question: str) -> dict:
    """
    Deterministic offline execution of the same two tools.

    This is a local verification path, not proof that CrewAI's own
    AgentExecutor invoked the tools during kickoff().
    """
    TOOL_EXECUTION_LOG.clear()
    TOOL_RESULT_LOG.clear()

    ticket_match = re.search(
        r"\bOLA-\d{4,}\b", question, re.IGNORECASE
    )
    ticket_result = None
    policy_result = None

    if ticket_match:
        ticket_id = ticket_match.group(0).upper()
        raw_ticket = ticket_status_tool.run(record_id=ticket_id)
        ticket_result = json.loads(raw_ticket)

    if not ticket_match or any(
        word in question.lower()
        for word in ("policy", "priority", "sla", "refund", "escalation")
    ):
        raw_policy = search_support_policies.run(question=question)
        policy_result = json.loads(raw_policy)

    if ticket_result and ticket_result.get("found"):
        ticket = ticket_result
        answer = (
            f"Ticket {ticket['record_id']} is currently {ticket['status']}. "
            f"Category: {ticket['category']}. "
            f"Resolution time: {ticket['resolution_time_hours']} hours. "
            f"Escalation score: {ticket['escalation_score']}."
        )
    elif ticket_result:
        answer = "Ticket not found. Please check the ticket ID."
    elif policy_result and not policy_result.get("fallback", False):
        answer = policy_result.get("answer", "")
    else:
        answer = "I don't know based on the available support policies."

    return {
        "question": question,
        "answer": answer,
        "ticket": ticket_result,
        "policy": policy_result,
        "tools_executed": list(TOOL_EXECUTION_LOG),
        "mode": "MOCK_LLM",
    }


if __name__ == "__main__":
    crew = build_support_crew()

    print("Three-agent crew constructed successfully.")
    print("Agents:", len(crew.agents))
    print("Tasks:", len(crew.tasks))
    print(
        "Configured tools:",
        [search_support_policies.name, ticket_status_tool.name],
    )

    # Use separate kickoff queries to show policy retrieval and ticket lookup.
    for question in (
        "What is the ticket priority policy?",
        "What is the status of ticket OLA-0001?",
    ):
        print(f"\nRunning CrewAI kickoff for: {question}")
        TOOL_EXECUTION_LOG.clear()
        TOOL_RESULT_LOG.clear()
        try:
            result = crew.kickoff(inputs={"question": question})
            validated = validate_crew_response(result)
            print("Pydantic-validated CrewAI response:")
            print(validated.model_dump_json(indent=2))
            print("CrewAI tool execution log:", TOOL_EXECUTION_LOG)
            assert validated.mode == "MOCK_LLM"
        except Exception as exc:
            print(f"CrewAI kickoff/validation failed: {type(exc).__name__}: {exc}")

    print("\nRunning explicit offline tool verification...")
    print(json.dumps(
        run_local_tools("What is the status of ticket OLA-0001?"),
        indent=2,
        ensure_ascii=False,
    ))
