"""Regression tests for cache, session carryover, and request budgets."""

import unittest

from mock_workflow import (
    MAX_INPUT_TOKENS,
    MAX_REQUEST_COST_USD,
    estimate_mock_cost_usd,
    get_workflow_metrics,
    reset_session,
    reset_workflow_metrics,
    response_cache,
    run_support_workflow,
)


class WorkflowControlTests(unittest.TestCase):
    def setUp(self):
        response_cache.clear()
        reset_workflow_metrics()
        reset_session("cache-test")
        reset_session("memory-test")
        reset_session("fresh-test")

    def test_normalized_cache_hit_avoids_second_ticket_tool_call(self):
        first = run_support_workflow(
            "What is the status of ticket OLA-0001?",
            session_id="cache-test",
        )
        after_first = get_workflow_metrics()
        second = run_support_workflow(
            "  WHAT IS THE STATUS OF TICKET ola-0001?  ",
            session_id="cache-test",
        )
        after_second = get_workflow_metrics()

        self.assertFalse(first.get("cache_hit", False))
        self.assertTrue(second.get("cache_hit", False))
        self.assertEqual(after_first["ticket_tool_calls"], 1)
        self.assertEqual(after_second["ticket_tool_calls"], 1)
        self.assertEqual(after_second["cache_hits"], 1)

    def test_session_context_carries_ticket_and_reset_removes_it(self):
        run_support_workflow(
            "What is the status of ticket OLA-0001?",
            session_id="memory-test",
        )
        follow_up = run_support_workflow(
            "What is its escalation score?",
            session_id="memory-test",
        )
        self.assertIn("Escalation score:", follow_up["answer"])
        self.assertIn("0.", follow_up["answer"])

        reset_session("memory-test")
        after_reset = run_support_workflow(
            "What is its escalation score?",
            session_id="memory-test",
        )
        self.assertTrue(after_reset["fallback"])
        self.assertIn("provide a ticket ID", after_reset["answer"])
        self.assertNotIn("Escalation score:", after_reset["answer"])

        fresh = run_support_workflow(
            "What is its escalation score?",
            session_id="fresh-test",
        )
        self.assertTrue(fresh["fallback"])
        self.assertIn("provide a ticket ID", fresh["answer"])

    def test_oversized_request_is_rejected_before_tools(self):
        response = run_support_workflow("x" * (MAX_INPUT_TOKENS * 4 + 20))
        self.assertTrue(response["fallback"])
        self.assertIn("input-token limit", response["answer"])
        metrics = get_workflow_metrics()
        self.assertEqual(metrics["rejected_oversized_requests"], 1)
        self.assertEqual(metrics["ticket_tool_calls"], 0)
        self.assertEqual(metrics["policy_tool_calls"], 0)

    def test_illustrative_cost_budget_is_enforced(self):
        token_count = int(MAX_REQUEST_COST_USD * 1000 / 0.001) + 1
        question = "x" * (token_count * 4)
        response = run_support_workflow(question)
        self.assertTrue(response["fallback"])
        self.assertIn("per-request budget", response["answer"])
        metrics = get_workflow_metrics()
        self.assertEqual(metrics["rejected_budget_requests"], 1)
        self.assertEqual(metrics["ticket_tool_calls"], 0)
        self.assertGreater(estimate_mock_cost_usd(token_count), MAX_REQUEST_COST_USD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
