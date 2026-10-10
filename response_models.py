
"""Pydantic response models for the Ola Support Agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class TicketDetails(BaseModel):
    """Validated support ticket information."""

    record_id: str
    status: str
    category: str
    resolution_time_hours: float = Field(ge=0)
    escalation_score: float = Field(ge=0, le=1)


class SupportResponse(BaseModel):
    """Structured and validated support-agent response."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    mode: Literal["MOCK_LLM"] = "MOCK_LLM"
    fallback: bool = False
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    ticket: TicketDetails | None = None


def validate_support_response(data: dict[str, Any]) -> SupportResponse:
    """Validate a response dictionary and return a typed model."""
    return SupportResponse.model_validate(data)


if __name__ == "__main__":
    sample = {
        "answer": "Ticket OLA-0001 is Escalated.",
        "mode": "MOCK_LLM",
        "fallback": False,
        "sources": [],
        "tools_used": ["check_support_ticket_status"],
        "ticket": {
            "record_id": "OLA-0001",
            "status": "Escalated",
            "category": "Billing",
            "resolution_time_hours": 91.62,
            "escalation_score": 0.173,
        },
    }

    validated = validate_support_response(sample)
    print("Pydantic validation: SUCCESS")
    print(validated.model_dump())
