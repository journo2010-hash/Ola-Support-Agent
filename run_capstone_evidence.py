"""Run deterministic, network-free acceptance demonstrations for capstone controls."""

import contextlib
import io
import time
from pathlib import Path

from mock_workflow import (
    MAX_INPUT_TOKENS,
    MAX_REQUEST_COST_USD,
    get_workflow_metrics,
    reset_session,
    reset_workflow_metrics,
    response_cache,
    run_support_workflow,
)


def main():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print("OLA SUPPORT AGENT — DETERMINISTIC CONTROL EVIDENCE")
        print("Mode: MOCK_LLM; ticket-only demonstrations; no model/API call.")
        print("\n1. NORMALIZED QUERY CACHE")
        response_cache.clear()
        reset_workflow_metrics()
        query = "What is the status of ticket OLA-0001?"
        started = time.perf_counter()
        first = run_support_workflow(query, session_id="cache-evidence")
        first_duration_ms = (time.perf_counter() - started) * 1000
        after_first = get_workflow_metrics()
        started = time.perf_counter()
        second = run_support_workflow(
            "  WHAT IS THE STATUS OF TICKET ola-0001?  ",
            session_id="cache-evidence",
        )
        second_duration_ms = (time.perf_counter() - started) * 1000
        after_second = get_workflow_metrics()
        print("First answer:", first["answer"])
        print(f"First request duration_ms: {first_duration_ms:.3f}")
        print("Counters after first request:", after_first)
        print("Second response cache_hit:", second.get("cache_hit", False))
        print(f"Second request duration_ms: {second_duration_ms:.3f}")
        print("Counters after second request:", after_second)
        assert second.get("cache_hit") is True
        assert after_second["ticket_tool_calls"] == after_first["ticket_tool_calls"] == 1
        assert after_second["cache_hits"] == 1
        print("PASS: repeated normalized query hit cache; no redundant ticket tool call.")

        print("\n2. SESSION MEMORY CARRYOVER AND RESET")
        reset_session("memory-evidence")
        first_turn = run_support_workflow(
            "What is the status of ticket OLA-0001?",
            session_id="memory-evidence",
        )
        follow_up = run_support_workflow(
            "What is its escalation score?",
            session_id="memory-evidence",
        )
        print("Turn 1:", first_turn["answer"])
        print("Turn 2:", follow_up["answer"])
        assert "Escalation score:" in follow_up["answer"]

        reset_session("memory-evidence")
        after_reset = run_support_workflow(
            "What is its escalation score?",
            session_id="memory-evidence",
        )
        fresh_session = run_support_workflow(
            "What is its escalation score?",
            session_id="fresh-evidence",
        )
        print("After reset:", after_reset["answer"])
        print("Fresh session:", fresh_session["answer"])
        assert after_reset["fallback"] and fresh_session["fallback"]
        print("PASS: ticket context carries across turns and disappears after reset.")

        print("\n3. INPUT TOKEN AND ILLUSTRATIVE COST BUDGETS")
        reset_workflow_metrics()
        oversized = run_support_workflow("x" * (MAX_INPUT_TOKENS * 4 + 20))
        print("Oversized request response:", oversized["answer"])
        print("Oversized request governance:", oversized["governance"])
        cost_tokens = int(MAX_REQUEST_COST_USD * 1000 / 0.001) + 1
        over_budget = run_support_workflow("x" * (cost_tokens * 4))
        print("Over-budget response:", over_budget["answer"])
        print("Over-budget governance:", over_budget["governance"])
        print("Budget counters:", get_workflow_metrics())
        assert oversized["fallback"] and over_budget["fallback"]
        assert get_workflow_metrics()["rejected_oversized_requests"] == 1
        assert get_workflow_metrics()["rejected_budget_requests"] == 1
        print(
            "NOTE: the displayed cost is a synthetic local budget unit used "
            "only to exercise the cap. MOCK_LLM makes no billable provider calls."
        )
        print("\nALL CONTROL DEMONSTRATIONS PASSED")

    transcript = output.getvalue()
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)
    target = evidence_dir / "capstone_controls_transcript.txt"
    target.write_text(transcript, encoding="utf-8")
    print(transcript, end="")
    print(f"Transcript saved to {target.as_posix()}")


if __name__ == "__main__":
    main()
