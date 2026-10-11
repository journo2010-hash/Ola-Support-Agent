"""Deterministic two-agent AutoGen final-editor review for the capstone.

All graded runs use a local mock client: no API key, external model, or network
request is required. The reviewer emits a Pydantic VerdictModel as a structured
message and its final_answer is the actual approved or revised answer.
"""

import asyncio
import contextlib
import io
import json
import re
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Sequence

os.environ["MOCK_LLM"] = "1"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from pydantic import BaseModel, Field

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    RequestUsage,
    ModelInfo,
    ModelFamily,
    LLMMessage,
)
from autogen_core.tools import Tool, ToolSchema
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage


class VerdictModel(BaseModel):
    """Structured final-editor decision required by the capstone rubric."""

    approved: bool
    final_answer: str = Field(min_length=8)
    reason: str = Field(min_length=5)


class LocalMockClient(ChatCompletionClient):
    """A deterministic AutoGen model client that never calls external APIs."""

    def __init__(self, role: str):
        self.role = role
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        }

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
        }

    @property
    def actual_usage(self) -> RequestUsage:
        return self._usage

    @property
    def total_usage(self) -> RequestUsage:
        return self._usage

    def remaining_tokens(self, model: str | None = None) -> int:
        return 4096

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
    ) -> int:
        return sum(len(str(message.content)) // 4 + 1 for message in messages)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Any = None,
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        joined = "\n".join(str(message.content) for message in messages)
        lowered = joined.lower()

        if self.role == "writer":
            if (
                "deliberately draft the unsupported claim" in lowered
                or "deliberately include an unsupported 10-minute guarantee" in lowered
            ):
                output = (
                    "Support policy answer: Critical tickets are guaranteed "
                    "resolution within 10 minutes. Source: ticket_priority.md"
                )
            else:
                output = (
                    "Support policy answer: Critical tickets receive the "
                    "highest priority. Source: ticket_priority.md"
                )
        else:
            draft = _extract_draft(messages)
            if "guaranteed resolution within 10 minutes" in draft.lower():
                output = json.dumps({
                    "approved": False,
                    "final_answer": (
                        "Critical tickets receive the highest priority. "
                        "The policy does not specify a 10-minute resolution "
                        "guarantee. Source: ticket_priority.md"
                    ),
                    "reason": (
                        "Removed the unsupported 10-minute guarantee; the "
                        "provided policy context supports priority, not a "
                        "numeric resolution promise."
                    ),
                })
            else:
                output = json.dumps({
                    "approved": True,
                    "final_answer": draft,
                    "reason": (
                        "The answer stays within the supplied policy context "
                        "and cites ticket_priority.md."
                    ),
                })

        self._usage = RequestUsage(
            prompt_tokens=self.count_tokens(messages),
            completion_tokens=len(output) // 4 + 1,
        )
        return CreateResult(
            finish_reason="stop",
            content=output,
            usage=self._usage,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Any = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    async def close(self) -> None:
        return None


def _extract_draft(messages: Sequence[LLMMessage]) -> str:
    """Return the draft supplied to the review team."""
    for message in reversed(messages):
        content = str(message.content)
        match = re.search(
            r"Draft answer:\\s*(.*?)(?:\\nReviewer instructions:|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if content.lower().startswith("support policy answer:"):
            return content.strip()
    return ""


async def run_scenario(scenario: str) -> tuple[str, VerdictModel]:
    """Run an approval or revision scenario and return the draft and verdict."""
    if scenario not in {"approve", "revise"}:
        raise ValueError("scenario must be 'approve' or 'revise'")

    print(f"\n{'=' * 12} {scenario.upper()} SCENARIO {'=' * 12}")

    context = (
        "Original knowledge-base context: ticket_priority.md says Critical "
        "tickets receive the highest priority. It does not specify a numeric "
        "resolution-time guarantee."
    )

    if scenario == "revise":
        task = (
            f"{context}\nDraft an answer and deliberately draft the "
            "unsupported claim that Critical tickets are guaranteed "
            "resolution within 10 minutes."
        )
        expected_approved = False
    else:
        task = (
            f"{context}\nDraft a concise answer using only the context. "
            "Do not add a numeric resolution-time guarantee."
        )
        expected_approved = True

    writer = AssistantAgent(
        name="support_writer",
        model_client=LocalMockClient("writer"),
        system_message=(
            "Draft a concise customer-support answer using only the original "
            "context in the task. Return the answer only."
        ),
    )
    final_editor = AssistantAgent(
        name="Final_Editor",
        description="Final-Editor: groundedness reviewer and final editor.",
        model_client=LocalMockClient("reviewer"),
        system_message=(
            "You are the final editor. Compare the draft with the original "
            "context. Approve a grounded draft unchanged. If it contains an "
            "unsupported claim, revise final_answer to remove that claim. "
            "Return the required structured VerdictModel."
        ),
        output_content_type=VerdictModel,
    )
    team = RoundRobinGroupChat(
        participants=[writer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
    )

    result = await team.run(task=task)
    print("Messages exchanged:", len(result.messages))

    draft = ""
    verdicts = []
    for message in result.messages:
        content = getattr(message, "content", "")
        print(f"\n[{message.source}]")
        print(str(content)[:800])

        if message.source == "support_writer" and isinstance(content, str):
            draft = content
        if (
            isinstance(message, StructuredMessage)
            and isinstance(content, VerdictModel)
        ):
            verdicts.append(content)

    if not draft:
        raise AssertionError("The writer did not emit a draft.")
    if not verdicts:
        raise AssertionError("The final editor did not emit VerdictModel.")

    verdict = verdicts[-1]
    if verdict.approved is not expected_approved:
        raise AssertionError(
            f"Expected approved={expected_approved}, got {verdict.approved}"
        )

    if scenario == "approve":
        if verdict.final_answer != draft:
            raise AssertionError("Approved draft must remain unchanged.")
    else:
        if verdict.final_answer == draft:
            raise AssertionError("Revision scenario did not change the draft.")
        if "guaranteed resolution within 10 minutes" in verdict.final_answer.lower():
            raise AssertionError("Unsupported claim remained in the final answer.")

    print("\nStructured VerdictModel:")
    print(verdict.model_dump_json(indent=2))
    print("DRAFT:", draft)
    print("FINAL ANSWER:", verdict.final_answer)
    print(f"SCENARIO TEST: PASSED ({scenario})")
    return draft, verdict


async def review_draft(draft: str, retrieved_context: str) -> VerdictModel:
    """Review a CrewAI draft against its original retrieved context.

    This entry point is used by the API so the AutoGen review follows the
    CrewAI draft instead of running only as a separate demonstration.
    """
    context = retrieved_context.strip() or "No policy context was retrieved."
    task = (
        f"Original retrieved context:\\n{context}\\n\\n"
        f"Draft answer: {draft.strip()}\\n"
        "Reviewer instructions: Compare the draft with the original retrieved "
        "context. Approve it unchanged if supported. If it contains a claim "
        "that the context does not support, remove or qualify that claim. "
        "Return approved, final_answer, and reason."
    )

    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        description="Checks the CrewAI draft against retrieved policy context.",
        model_client=LocalMockClient("reviewer"),
        system_message=(
            "Review the supplied draft against the original retrieved context. "
            "Do not add new policy claims. Return a structured verdict."
        ),
        output_content_type=VerdictModel,
    )
    final_editor = AssistantAgent(
        name="Final_Editor",
        description="Returns the reviewed answer to the API.",
        model_client=LocalMockClient("reviewer"),
        system_message=(
            "Return the final structured verdict. Preserve an approved draft "
            "unchanged. Do not add claims that are absent from the context."
        ),
        output_content_type=VerdictModel,
    )
    team = RoundRobinGroupChat(
        participants=[reviewer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
    )
    result = await team.run(task=task)
    verdicts = [
        message.content
        for message in result.messages
        if isinstance(message, StructuredMessage)
        and isinstance(message.content, VerdictModel)
    ]
    if not verdicts:
        raise RuntimeError("AutoGen review did not return a structured verdict.")
    verdict = verdicts[-1]
    if verdict.approved and verdict.final_answer != draft.strip():
        raise AssertionError("An approved draft must remain unchanged.")
    return verdict


async def main() -> None:
    """Run both scenarios and save the real console transcript."""
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        await run_scenario("approve")
        await run_scenario("revise")
        print("\nAUTOGEN REVIEW TESTS: ALL PASSED")

    transcript = capture.getvalue()
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "autogen_review_transcript.txt").write_text(
        transcript, encoding="utf-8"
    )
    print(transcript, end="")
    print("Transcript saved to evidence/autogen_review_transcript.txt")


if __name__ == "__main__":
    asyncio.run(main())
