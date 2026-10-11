"""Regression tests for the output-groundedness guard."""

import unittest

from guardrails import enforce_output_grounding


class OutputGroundingTests(unittest.TestCase):
    def test_non_fallback_without_sources_is_rejected(self):
        result = enforce_output_grounding(
            "A claim that has no source.",
            [],
            fallback=False,
        )
        self.assertTrue(result["fallback"])
        self.assertFalse(result["grounded"])
        self.assertEqual(result["sources"], [])
        self.assertIn("I don't know", result["answer"])

    def test_non_fallback_with_source_is_accepted(self):
        result = enforce_output_grounding(
            "Relevant policy information.",
            ["sla_policy.md"],
            fallback=False,
        )
        self.assertFalse(result["fallback"])
        self.assertTrue(result["grounded"])
        self.assertEqual(result["sources"], ["sla_policy.md"])

    def test_existing_fallback_stays_fallback(self):
        result = enforce_output_grounding(
            "I don't know based on the available support policies.",
            [],
            fallback=True,
        )
        self.assertTrue(result["fallback"])
        self.assertFalse(result["grounded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
