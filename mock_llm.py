
"""Deterministic local mock LLM with CrewAI tool-call support."""

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM


class MockSupportLLM(BaseLLM):
    """Local deterministic LLM for offline CrewAI testing."""

    def __init__(self):
        super().__init__(
            model="mock-support-local",
            temperature=0,
        )
        self._call_counter = 0

    def supports_function_calling(self) -> bool:
        """Enable CrewAI's native tool-calling path."""
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    @staticmethod
    def _message_text(messages: Any) -> str:
        """Convert CrewAI messages into readable text."""
        if isinstance(messages, str):
            return messages

        parts = []

        for message in messages or []:
            if isinstance(message, dict):
                content = message.get("content", "")
                if content:
                    parts.append(str(content))
            else:
                content = getattr(message, "content", None)
                parts.append(str(content if content is not None else message))

        return "\n".join(parts)

    @staticmethod
    def _tool_names(tools: list | None) -> dict[str, str]:
        """Map normalized tool names to their actual schema names."""
        names = {}

        for item in tools or []:
            if not isinstance(item, dict):
                continue

            function = item.get("function", {})
            name = function.get("name") or item.get("name")

            if name:
                names[name.lower()] = name

        return names

    @staticmethod
    def _extract_question(
        text: str,
        from_task: Any = None,
    ) -> str:
        """Extract the user's question from the current task."""
        description = getattr(from_task, "description", "") or ""

        # Prefer the task description because it usually contains the
        # current question without unrelated conversation history.
        patterns = [
            r"this question:\s*(.+?)\.\s*Use the search_support_policies",
            r"Question:\s*(.+?)\.\s*Report the returned ticket data",
            r"Answer this question:\s*(.+?)\.\s*Use only the verified",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                description,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()

        # Fallback for task templates that differ between CrewAI versions.
        match = re.search(
            r"\b(?:question|request)\s*:\s*(.+)",
            description,
            re.IGNORECASE | re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return description.strip() or text[-1000:]

    @staticmethod
    def _has_tool_result(messages: Any) -> bool:
        """Check whether CrewAI has already supplied a tool observation."""
        if not isinstance(messages, list):
            return False

        for message in messages:
            if not isinstance(message, dict):
                continue

            if message.get("role") == "tool":
                return True

        return False

    @staticmethod
    def _last_tool_result(messages: Any) -> str:
        """Return the latest native tool result, if present."""
        if not isinstance(messages, list):
            return ""

        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "tool"
            ):
                return str(message.get("content", ""))

        return ""

    def _make_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return a native function-call response understood by CrewAI."""
        self._call_counter += 1

        return [
            {
                "id": f"mock-call-{self._call_counter}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ]

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        """Execute deterministic local reasoning and tool selection."""
        text = self._message_text(messages)

        # Keep structured-output support for compatible tasks.
        if response_model is not None:
            fields = getattr(response_model, "model_fields", {})
            values = {}

            for name, field in fields.items():
                annotation = field.annotation

                if annotation is bool:
                    values[name] = True
                elif annotation is int:
                    values[name] = 1
                elif annotation is float:
                    values[name] = 1.0
                else:
                    values[name] = (
                        "REVISE"
                        if "verdict" in name.lower()
                        else "Mock local response; verify against evidence."
                    )

            return response_model(**values)

        role = str(getattr(from_agent, "role", "") or "").lower()
        task_description = str(
            getattr(from_task, "description", "") or ""
        ).lower()

        tool_names = self._tool_names(tools)
        question = self._extract_question(text, from_task)

        # If this agent already called a tool, return its observation
        # rather than repeatedly issuing the same tool call.
        if self._has_tool_result(messages):
            observation = self._last_tool_result(messages)

            if "retrieval" in role:
                return (
                    "Local policy retrieval completed. "
                    f"Verified tool result: {observation}"
                )

            if "lookup" in role:
                return (
                    "Local ticket lookup completed. "
                    f"Verified tool result: {observation}"
                )

            return (
                "Verified local tool result received: "
                f"{observation}"
            )

        # Retrieval agent: invoke the policy search tool.
        search_name = tool_names.get("search_support_policies")

        if search_name and (
            "retrieval" in role
            or "search_support_policies" in task_description
        ):
            return self._make_tool_call(
                search_name,
                {"question": question},
            )

        # Lookup agent: invoke ticket lookup only when an ID is present.
        lookup_name = tool_names.get("check_support_ticket_status")

        if lookup_name and (
            "lookup" in role
            or "check_support_ticket_status" in task_description
        ):
            ticket_match = re.search(
                r"\bOLA-\d{4,}\b",
                text,
                re.IGNORECASE,
            )

            if ticket_match:
                return self._make_tool_call(
                    lookup_name,
                    {"record_id": ticket_match.group(0).upper()},
                )

            return "No ticket ID was supplied, so no ticket lookup was performed."

        # Composer agent: produce a local response based on prior evidence.
        if "composer" in role:
            ticket_match = re.search(
                r"\bOLA-\d{4,}\b",
                text,
                re.IGNORECASE,
            )

            if ticket_match:
                return (
                    "The question concerns ticket "
                    f"{ticket_match.group(0).upper()}. "
                    "Use the verified ticket lookup result in the preceding "
                    "task context; do not invent missing ticket details."
                )

            return (
                "Use the policy evidence and source documents returned by "
                "the retrieval task. Summarize only verified evidence and "
                "state when the evidence is insufficient."
            )

        # General deterministic response when no tool is applicable.
        return (
            "MOCK_LLM: no external model was called. "
            "No applicable local tool was selected for this task."
        )
