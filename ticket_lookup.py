
"""Ticket lookup and escalation scoring for the Ola Support Agent capstone."""

from dataset import SUPPORT_TICKETS


def check_support_ticket_status(record_id: str) -> dict:
    """Return ticket status, resolution time, and an escalation score."""

    ticket = next(
        (
            item
            for item in SUPPORT_TICKETS
            if item["record_id"].lower() == record_id.strip().lower()
        ),
        None,
    )

    if ticket is None:
        return {
            "found": False,
            "record_id": record_id,
            "message": "Ticket not found. Please check the ticket ID.",
        }

    # Recent tickets receive a higher recency score.
    recency_score = max(
        0.0,
        min(1.0, 1.0 - ticket["days_since_created"] / 30.0),
    )

    # Escalation contributes 60%; recency contributes 40%.
    escalation_score = (
        0.6 * float(ticket["escalated"])
        + 0.4 * recency_score
    )

    return {
        "found": True,
        "record_id": ticket["record_id"],
        "category": ticket["category"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created": ticket["days_since_created"],
        "escalated": ticket["escalated"],
        "escalation_score": round(escalation_score, 3),
    }


if __name__ == "__main__":
    print("Ticket lookup test:")
    print(check_support_ticket_status("OLA-0001"))
    print("\nUnknown ticket test:")
    print(check_support_ticket_status("OLA-9999"))
