
"""Generate a deterministic synthetic Ola support-ticket dataset."""

import random
from collections import Counter

BASE_SEED = 42
NUMBER_OF_TICKETS = 50

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]

STATUSES = [
    "Open",
    "In Progress",
    "Escalated",
    "Resolved",
    "Closed",
]


def generate_tickets(seed=BASE_SEED, count=NUMBER_OF_TICKETS):
    """Generate reproducible tickets with consistent escalation data."""

    if count < 40:
        raise ValueError("The dataset must contain at least 40 tickets.")

    for actual_seed in range(seed, seed + 10_000):
        rng = random.Random(actual_seed)

        # Balanced category coverage.
        categories = (CATEGORIES * ((count + 4) // 5))[:count]
        rng.shuffle(categories)

        # Generate escalation flags using a weighted random distribution.
        escalation_flags = [
            rng.choices([True, False], weights=[20, 80], k=1)[0]
            for _ in range(count)
        ]

        escalation_count = sum(escalation_flags)
        escalation_pct = 100 * escalation_count / count

        # Retry with the next seed if escalation is outside the required range.
        if not 10 <= escalation_pct <= 30:
            continue

        # Ensure status and escalation flag always agree.
        other_statuses = [
            "Open",
            "In Progress",
            "Resolved",
            "Closed",
        ]

        statuses = []
        other_status_index = 0

        for is_escalated in escalation_flags:
            if is_escalated:
                statuses.append("Escalated")
            else:
                statuses.append(
                    other_statuses[
                        other_status_index % len(other_statuses)
                    ]
                )
                other_status_index += 1

        # Shuffle statuses and flags together to avoid an ordered pattern.
        records = list(zip(statuses, escalation_flags))
        rng.shuffle(records)

        statuses = [record[0] for record in records]
        escalation_flags = [record[1] for record in records]

        tickets = []

        for i in range(count):
            tickets.append({
                "record_id": f"OLA-{i + 1:04d}",
                "category": categories[i],
                "status": statuses[i],
                "resolution_time_hours": round(
                    rng.uniform(0.5, 120.0), 2
                ),
                "days_since_created": rng.randint(0, 30),
                "escalated": escalation_flags[i],
            })

        return tickets, actual_seed

    raise RuntimeError("Could not generate a valid dataset.")


def validate_tickets(tickets):
    """Validate required fields and dataset acceptance criteria."""

    required_fields = {
        "record_id",
        "category",
        "status",
        "resolution_time_hours",
        "days_since_created",
        "escalated",
    }

    assert len(tickets) >= 40
    assert all(set(ticket) == required_fields for ticket in tickets)
    assert len({ticket["record_id"] for ticket in tickets}) == len(tickets)

    assert all(ticket["category"] in CATEGORIES for ticket in tickets)
    assert all(ticket["status"] in STATUSES for ticket in tickets)

    assert all(
        0 <= ticket["days_since_created"] <= 30
        for ticket in tickets
    )

    assert all(
        0.5 <= ticket["resolution_time_hours"] <= 120.0
        for ticket in tickets
    )

    assert all(
        isinstance(ticket["escalated"], bool)
        for ticket in tickets
    )

    # Critical consistency check: status and escalation flag must agree.
    assert all(
        (ticket["status"] == "Escalated") == ticket["escalated"]
        for ticket in tickets
    ), "Ticket status and escalation flag are inconsistent."

    category_counts = Counter(
        ticket["category"] for ticket in tickets
    )

    status_counts = Counter(
        ticket["status"] for ticket in tickets
    )

    assert all(
        category_counts[category] >= 3
        for category in CATEGORIES
    )

    assert all(
        status_counts[status] >= 1
        for status in STATUSES
    )

    escalated_pct = (
        100 * sum(ticket["escalated"] for ticket in tickets)
        / len(tickets)
    )

    assert 10 <= escalated_pct <= 30

    return category_counts, status_counts, escalated_pct


# Generate the dataset once when the module is loaded.
SUPPORT_TICKETS, USED_SEED = generate_tickets()


if __name__ == "__main__":
    category_counts, status_counts, escalated_pct = validate_tickets(
        SUPPORT_TICKETS
    )

    print("DATASET VALIDATION: PASSED")
    print(f"Base seed: {BASE_SEED}")
    print(f"Seed used: {USED_SEED}")
    print(f"Total tickets: {len(SUPPORT_TICKETS)}")
    print(f"Category counts: {dict(category_counts)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Escalated percentage: {escalated_pct:.2f}%")
    print(
        "Resolution time range: 0.5–120 hours; "
        "chosen to represent short and longer support cases."
    )

    print("\nFirst five tickets:")
    for ticket in SUPPORT_TICKETS[:5]:
        print(ticket)

    print("\nESCALATION CONSISTENCY CHECK: PASSED")
