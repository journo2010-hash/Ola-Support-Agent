
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
    """Generate reproducible records and retry seeds if escalation is out of range."""

    if count < 40:
        raise ValueError("The dataset must contain at least 40 tickets.")

    for actual_seed in range(seed, seed + 10_000):
        rng = random.Random(actual_seed)

        # Balanced category and status coverage; shuffle reproducibly.
        categories = (CATEGORIES * ((count + 4) // 5))[:count]
        statuses = (STATUSES * ((count + 4) // 5))[:count]
        rng.shuffle(categories)
        rng.shuffle(statuses)

        # Weighted random generation: approximately 20% escalated.
        escalation_flags = [
            rng.choices([True, False], weights=[20, 80], k=1)[0]
            for _ in range(count)
        ]

        escalation_count = sum(escalation_flags)
        escalation_pct = 100 * escalation_count / count

        # Regenerate with a new seed if the percentage misses the brief.
        if not 10 <= escalation_pct <= 30:
            continue

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
    """Check required fields and dataset acceptance criteria."""

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
    assert len({t["record_id"] for t in tickets}) == len(tickets)
    assert all(t["category"] in CATEGORIES for t in tickets)
    assert all(t["status"] in STATUSES for t in tickets)
    assert all(0 <= t["days_since_created"] <= 30 for t in tickets)
    assert all(t["resolution_time_hours"] > 0 for t in tickets)
    assert all(isinstance(t["escalated"], bool) for t in tickets)

    category_counts = Counter(t["category"] for t in tickets)
    status_counts = Counter(t["status"] for t in tickets)

    assert all(category_counts[c] >= 3 for c in CATEGORIES)
    assert all(status_counts[s] >= 1 for s in STATUSES)

    escalated_pct = (
        100 * sum(t["escalated"] for t in tickets) / len(tickets)
    )
    assert 10 <= escalated_pct <= 30

    return category_counts, status_counts, escalated_pct



# Generate once when the module is loaded.
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

