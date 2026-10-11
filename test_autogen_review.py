"""Tests for AutoGen's structured approval and actual revision behavior."""

import asyncio
import unittest

from autogen_review import VerdictModel, review_draft, run_scenario


class AutoGenReviewTests(unittest.TestCase):
    def test_approval_keeps_grounded_draft_unchanged(self):
        draft, verdict = asyncio.run(run_scenario("approve"))
        self.assertIsInstance(verdict, VerdictModel)
        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.final_answer, draft)


    def test_review_draft_approves_grounded_crew_answer_unchanged(self):
        draft = "The policy gives Critical tickets the highest priority."
        context = "ticket_priority.md: Critical tickets receive the highest priority."
        verdict = asyncio.run(review_draft(draft, context))
        self.assertIsInstance(verdict, VerdictModel)
        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.final_answer, draft)

    def test_review_draft_revises_known_unsupported_claim(self):
        draft = (
            "Critical tickets are guaranteed resolution within 10 minutes."
        )
        context = (
            "ticket_priority.md: Critical tickets receive the highest priority. "
            "No resolution-time guarantee is stated."
        )
        verdict = asyncio.run(review_draft(draft, context))
        self.assertIsInstance(verdict, VerdictModel)
        self.assertFalse(verdict.approved)
        self.assertNotIn(
            "guaranteed resolution within 10 minutes",
            verdict.final_answer.lower(),
        )

    def test_revision_removes_deliberately_ungrounded_claim(self):
        draft, verdict = asyncio.run(run_scenario("revise"))
        self.assertIsInstance(verdict, VerdictModel)
        self.assertFalse(verdict.approved)
        self.assertNotEqual(verdict.final_answer, draft)
        self.assertNotIn(
            "guaranteed resolution within 10 minutes",
            verdict.final_answer.lower(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
