"""Contract test for the CrewAI-backed API route.

The CI environment intentionally does not install the full CrewAI/RAG stack,
so this test substitutes small local doubles and verifies the endpoint wiring.
"""

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


class FakeVerdict:
    def __init__(self, answer):
        self.approved = True
        self.final_answer = answer
        self.reason = "The supplied draft is supported by the retrieved context."

    def model_dump(self):
        return {
            "approved": self.approved,
            "final_answer": self.final_answer,
            "reason": self.reason,
        }


class FakeCrew:
    def kickoff(self, inputs):
        self.inputs = inputs
        return "Critical tickets receive the highest priority. Source: ticket_priority.md"


class CrewApiContractTests(unittest.TestCase):
    def test_ask_crew_runs_draft_through_review_and_returns_schema(self):
        execution_log = ["search_support_policies"]
        result_log = {
            "policy": {
                "answer": "Critical tickets receive the highest priority.",
                "sources": ["ticket_priority.md"],
                "fallback": False,
            },
            "ticket": {},
        }
        crew_module = types.ModuleType("crew_support")
        crew_module.TOOL_EXECUTION_LOG = execution_log
        crew_module.TOOL_RESULT_LOG = result_log
        crew_module.build_support_crew = FakeCrew
        crew_module.validate_crew_response = lambda raw: types.SimpleNamespace(
            answer=raw,
            mode="MOCK_LLM",
            fallback=False,
            sources=["ticket_priority.md"],
            tools_used=list(execution_log),
            ticket=None,
        )

        async def fake_review_draft(draft, context):
            self.assertIn("Critical tickets receive the highest priority.", context)
            return FakeVerdict(draft)

        review_module = types.ModuleType("autogen_review")
        review_module.review_draft = fake_review_draft

        with patch.dict(
            sys.modules,
            {
                "crew_support": crew_module,
                "autogen_review": review_module,
            },
        ):
            response = TestClient(app).post(
                "/ask-crew",
                json={"question": "What is the ticket priority policy?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "MOCK_LLM")
        self.assertEqual(body["sources"], ["ticket_priority.md"])
        self.assertTrue(body["review"]["approved"])
        self.assertIn("Critical tickets receive the highest priority.", body["answer"])
        self.assertEqual(body["tools_used"], ["search_support_policies"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
