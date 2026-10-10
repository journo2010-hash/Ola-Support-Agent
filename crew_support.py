
"""Three-agent CrewAI team with RAG and ticket lookup tools."""

import os
import json

os.environ["MOCK_LLM"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from mock_llm import MockSupportLLM
from ticket_lookup import check_support_ticket_status


# Cache the RAG collection so it is initialized only once per process.
_rag_components = None


def get_rag_components():
    global _rag_components
    if _rag_components is None:
        from rag import initialize_support_rag
        _rag_components = initialize_support_rag()
    return _rag_components


@tool("search_support_policies")
def search_support_policies(question: str) -> str:
    """Search support policy documents and return grounded evidence and sources."""
    from rag import grounded_answer

    collection, model, threshold = get_rag_components()
    result = grounded_answer(question, collection, model, threshold)
    return json.dumps(result, ensure_ascii=False)


@tool("check_support_ticket_status")
def ticket_status_tool(record_id: str) -> str:
    """Look up a support ticket using its ID, such as OLA-0001."""
    return json.dumps(
        check_support_ticket_status(record_id),
        ensure_ascii=False,
    )


def build_support_crew():
    """Create the three-agent support team."""

    retrieval_agent = Agent(
        role="Support Policy Retrieval Specialist",
        goal="Retrieve relevant evidence from the support policy knowledge base.",
        backstory="You use the policy retrieval tool and do not invent policies.",
        llm=MockSupportLLM(),
        tools=[search_support_policies],
        verbose=False,
        allow_delegation=False,
    )

    lookup_agent = Agent(
        role="Support Ticket Lookup Specialist",
        goal="Report ticket status and escalation details accurately.",
        backstory="You use the ticket lookup tool and never invent ticket data.",
        llm=MockSupportLLM(),
        tools=[ticket_status_tool],
        verbose=False,
        allow_delegation=False,
    )

    composer_agent = Agent(
        role="Support Response Composer",
        goal="Prepare a clear response using verified evidence only.",
        backstory="You state when the available evidence is insufficient.",
        llm=MockSupportLLM(),
        verbose=False,
        allow_delegation=False,
    )

    retrieval_task = Task(
        description=(
            "Search the policy knowledge base for evidence relevant to "
            "this question: {question}. Use the search_support_policies tool. "
            "Report the evidence and sources returned by the tool."
        ),
        expected_output="Retrieved policy evidence, sources, and fallback status.",
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=(
            "If the question contains a ticket ID, use the "
            "check_support_ticket_status tool. Question: {question}. "
            "Report the returned ticket data without changing it."
        ),
        expected_output="Verified ticket details, or state that no ticket ID was supplied.",
        agent=lookup_agent,
    )

    composer_task = Task(
        description=(
            "Answer this question: {question}. Use only the verified policy "
            "evidence and ticket details from the preceding tasks. If the "
            "evidence does not answer the question, say you do not know."
        ),
        expected_output="A concise response grounded in the preceding task results.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )


if __name__ == "__main__":
    crew = build_support_crew()
    print("Three-agent crew constructed successfully.")
    print("Agents:", len(crew.agents))
    print("Tasks:", len(crew.tasks))
    print("Retrieval tool:", search_support_policies.name)
    print("Lookup tool:", ticket_status_tool.name)
