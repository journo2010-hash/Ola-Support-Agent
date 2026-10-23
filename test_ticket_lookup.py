"""Regression tests for dataset-calibrated escalation scoring."""

import unittest

from dataset import SUPPORT_TICKETS
from ticket_lookup import (
    ESCALATION_SCORE_THRESHOLD,
    check_support_ticket_status,
)


class TicketLookupTests(unittest.TestCase):
    def test_score_and_threshold_are_bounded(self):
        for ticket in SUPPORT_TICKETS:
            result = check_support_ticket_status(ticket["record_id"])
            self.assertTrue(result["found"])
            self.assertGreaterEqual(result["escalation_score"], 0.0)
            self.assertLessEqual(result["escalation_score"], 1.0)
            self.assertGreaterEqual(result["escalation_threshold"], 0.0)
            self.assertLessEqual(result["escalation_threshold"], 1.0)
            self.assertEqual(
                result["priority_review_flag"],
                result["escalation_score"] >= result["escalation_threshold"],
            )

    def test_score_formula_and_threshold_method_are_reported(self):
        result = check_support_ticket_status(SUPPORT_TICKETS[0]["record_id"])
        self.assertIn("0.60 * escalated_boolean", result["score_formula"])
        self.assertIn("80th percentile", result["threshold_method"])
        self.assertAlmostEqual(
            result["escalation_threshold"],
            round(ESCALATION_SCORE_THRESHOLD, 3),
        )

    def test_unknown_ticket_returns_not_found(self):
        result = check_support_ticket_status("OLA-999999")
        self.assertFalse(result["found"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
