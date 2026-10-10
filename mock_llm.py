
"""A deterministic local mock LLM for CrewAI testing."""

from crewai.llms.base_llm import BaseLLM


class MockSupportLLM(BaseLLM):
    """Simple local model that returns deterministic responses."""

    def __init__(self):
        super().__init__(
            model="mock-support-local",
            temperature=0,
        )

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
        """Return a local response without any API call."""

        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(str(message) for message in messages)

        # Return a valid JSON object when structured output is requested.
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
                        "Mock review: revise if evidence is missing."
                        if "verdict" in name.lower()
                        else "Mock local response; verify against retrieved evidence."
                    )

            return response_model(**values)

        # This deliberately avoids claiming that tools were executed.
        return (
            "MOCK_LLM response: no external model was called. "
            "Use the verified output of the local support tools as evidence. "
            f"Input received ({len(text)} characters)."
        )
