"""Ticket lookup and dataset-calibrated escalation scoring.

The score is a transparent triage heuristic, not a calibrated probability.
"""

from math import ceil

from dataset import SUPPORT_TICKETS


ESCALATION_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40
MAX_AGE_DAYS = 30


def _recency_score(days_since_created: int) -> float:
    """Normalize ticket recency to [0, 1]; newer tickets score higher."""
    return max(0.0, min(1.0, 1.0 - days_since_created / MAX_AGE_DAYS))


def _raw_escalation_score(ticket: dict) -> float:
    """Formula: 0.60 * escalated + 0.40 * normalized recency."""
    return (
        ESCALATION_WEIGHT * float(ticket["escalated"])
        + RECENCY_WEIGHT * _recency_score(ticket["days_since_created"])
    )


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for non-empty data."""
    if not values:
        raise ValueError("Cannot calculate a percentile for empty data")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


# Dataset-calibrated threshold: nearest-rank 80th percentile of this
# synthetic dataset's own score distribution. Ties can flag slightly more
# than 20% of records. It is a triage cutoff, not a predicted probability.
_DATASET_SCORES = [_raw_escalation_score(ticket) for ticket in SUPPORT_TICKETS]
ESCALATION_SCORE_THRESHOLD = _nearest_rank_percentile(_DATASET_SCORES, 0.80)


def check_support_ticket_status(record_id: str) -> dict:
    """Return ticket status, resolution time, and escalation score."""
    normalized_id = record_id.strip().lower()
    ticket = next(
        (
            item
            for item in SUPPORT_TICKETS
            if item["record_id"].lower() == normalized_id
        ),
        None,
    )

    if ticket is None:
        return {
            "found": False,
            "record_id": record_id,
            "message": "Ticket not found. Please check the ticket ID.",
        }

    score = _raw_escalation_score(ticket)
    return {
        "found": True,
        "record_id": ticket["record_id"],
        "category": ticket["category"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created": ticket["days_since_created"],
        "escalated": ticket["escalated"],
        "escalation_score": round(score, 3),
        "escalation_threshold": round(ESCALATION_SCORE_THRESHOLD, 3),
        "priority_review_flag": score >= ESCALATION_SCORE_THRESHOLD,
        "score_formula": (
            "0.60 * escalated_boolean + "
            "0.40 * (1 - days_since_created / 30)"
        ),
        "threshold_method": (
            "nearest-rank 80th percentile of generated dataset scores"
        ),
    }


if __name__ == "__main__":
    flagged = sum(
        _raw_escalation_score(ticket) >= ESCALATION_SCORE_THRESHOLD
        for ticket in SUPPORT_TICKETS
    )
    print("Escalation scoring formula:")
    print("score = 0.60 * escalated + 0.40 * (1 - days_since_created / 30)")
    print(f"Dataset tickets: {len(SUPPORT_TICKETS)}")
    print(f"80th-percentile threshold: {ESCALATION_SCORE_THRESHOLD:.3f}")
    print(f"Tickets flagged for priority review: {flagged}/{len(SUPPORT_TICKETS)}")
    print("\nTicket lookup test:")
    print(check_support_ticket_status("OLA-0001"))
    print("\nUnknown ticket test:")
    print(check_support_ticket_status("OLA-9999"))
