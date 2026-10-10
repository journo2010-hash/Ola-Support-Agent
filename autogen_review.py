
import asyncio
import json
import os
from typing import Any, AsyncGenerator, Sequence, Literal

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
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage


# ==================================================
# 1. Structured Pydantic verdict
# ==================================================

class Verdict(BaseModel):
    decision: Literal["APPROVE", "REVISE"]
    reason: str = Field(min_length=5)
    suggested_revision: str | None = None


# ==================================================
# 2. Local deterministic mock model client
# ==================================================

class LocalMockClient(ChatCompletionClient):
    """Local deterministic client; does not call external APIs."""

    def __init__(self, role: str):
        self.role = role
        self._usage = RequestUsage(
            prompt_tokens=0,
            completion_tokens=0,
        )

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
        return sum(
            len(str(message.content)) // 4 + 1
            for message in messages
        )

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

        # The writer produces an answer based on the task.
        if self.role == "writer":
            prompt = "\n".join(
                str(message.content) for message in messages
            ).lower()

            if "missing its source" in prompt:
                output = (
                    "Support policy answer: tickets follow the documented "
                    "priority policy, but a source citation is missing."
                )
            else:
                output = (
                    "Support policy answer: follow the documented ticket "
                    "priority policy. Source: ticket_priority.md"
                )

        # The reviewer checks the writer's answer only.
        else:
            writer_answer = next(
                (
                    str(message.content).lower()
                    for message in reversed(messages)
                    if "support policy answer:" in str(message.content).lower()
                ),
                "",
            )

            if "source citation is missing" in writer_answer:
                output = json.dumps({
                    "decision": "REVISE",
                    "reason": (
                        "The answer needs a supporting knowledge-base citation."
                    ),
                    "suggested_revision": (
                        "Add a relevant knowledge-base source."
                    ),
                })
            else:
                output = json.dumps({
                    "decision": "APPROVE",
                    "reason": (
                        "The answer includes policy information and a source."
                    ),
                    "suggested_revision": None,
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

        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

        yield result

    async def close(self) -> None:
        return None


# ==================================================
# 3. Run one two-agent review scenario
# ==================================================

async def run_scenario(scenario: str) -> None:
    print(f"\n{'=' * 12} {scenario.upper()} SCENARIO {'=' * 12}")

    writer = AssistantAgent(
        name="support_writer",
        model_client=LocalMockClient("writer"),
        system_message=(
            "Draft a concise customer support answer about ticket priority. "
            "Return the answer only."
        ),
    )

    reviewer = AssistantAgent(
        name="support_reviewer",
        model_client=LocalMockClient("reviewer"),
        system_message=(
            "Review the support writer's answer. Produce a structured Verdict. "
            "APPROVE answers with a source; REVISE answers missing a source."
        ),
        output_content_type=Verdict,
    )

    team = RoundRobinGroupChat(
        participants=[writer, reviewer],
        termination_condition=MaxMessageTermination(3),
        custom_message_types=[StructuredMessage[Verdict]],
    )

    if scenario == "revise":
        task = (
            "Write a support answer about ticket priority. "
            "The answer is missing its source and must be revised."
        )
        expected = "REVISE"
    else:
        task = (
            "Write a support answer about ticket priority. "
            "The answer has a supporting source and should be approved."
        )
        expected = "APPROVE"

    result = await team.run(task=task)

    print("Messages exchanged:", len(result.messages))

    for message in result.messages:
        print(f"\n[{message.source}]")
        print(str(getattr(message, "content", ""))[:800])

    verdicts = [
        message.content
        for message in result.messages
        if isinstance(message, StructuredMessage)
        and isinstance(message.content, Verdict)
    ]

    if not verdicts:
        raise RuntimeError(
            "No structured Verdict was emitted by the reviewer."
        )

    verdict = verdicts[-1]

    print("\nStructured verdict:")
    print(verdict.model_dump_json(indent=2))

    if verdict.decision != expected:
        raise AssertionError(
            f"Expected {expected}, got {verdict.decision}"
        )

    print(f"SCENARIO TEST: PASSED ({expected})")


# ==================================================
# 4. Run both scenarios
# ==================================================

async def main() -> None:
    await run_scenario("approve")
    await run_scenario("revise")

    print("\nAUTOGEN REVIEW TESTS: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
