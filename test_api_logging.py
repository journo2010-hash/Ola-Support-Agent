"""API regression tests for per-request JSONL logs and phone privacy."""

import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import LOG_FILE, app


class ApiLoggingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.log_path = Path(LOG_FILE)
        self.before = (
            len(self.log_path.read_text(encoding="utf-8").splitlines())
            if self.log_path.exists()
            else 0
        )

    def test_each_http_request_has_one_trace_log(self):
        responses = [
            self.client.get("/"),
            self.client.post("/ask", json={}),  # Pydantic 422 still logged
            self.client.post(
                "/ask",
                json={"question": "Please help, my phone is 9876543210"},
            ),
        ]
        for response in responses:
            self.assertIn("X-Trace-ID", response.headers)

        safe_response = responses[2].json()
        self.assertIn("[PHONE REDACTED]", safe_response["question"])
        self.assertNotIn("9876543210", safe_response["question"])

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        new_events = [json.loads(line) for line in lines[self.before:]]
        self.assertEqual(len(new_events), 3)
        self.assertEqual(len({event["trace_id"] for event in new_events}), 3)
        self.assertTrue(all("duration_ms" in event for event in new_events))
        self.assertTrue(all("question" not in event for event in new_events))
        self.assertNotIn("9876543210", json.dumps(new_events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
