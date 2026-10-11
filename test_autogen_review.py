"""Tests for AutoGen's structured approval and actual revision behavior."""

import asyncio
import unittest

from autogen_review import VerdictModel, run_scenario


class AutoGenReviewTests(unittest.TestCase):
    def test_approval_keeps_grounded_draft_unchanged(self):
        draft, verdict = asyncio.run(run_scenario("approve"))
        self.assertIsInstance(verdict, VerdictModel)
        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.final_answer, draft)

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
