# Capstone rubric verification checklist

This file maps the institute rubric to the implementation and reproducible evidence. Run all commands from the repository root using the project's Python 3.11 virtual environment. The knowledge base and ticket records are synthetic demonstration data, not official Ola policy or real customer data.

## Part 1 — Dataset, knowledge base, and RAG

- [x] `dataset.py` creates 50 deterministic synthetic tickets with a base seed of 42, balanced categories, all required statuses, a 10–30% accepted escalation rate, ages from 0–30 days, and resolution times from 0.5–120 hours.
- [x] README documents category weights, status assignment policy, escalation weights, retry rule, seed, and resolution range.
- [x] `knowledge_base/` contains 12 Markdown documents, each with 2–4 policy sentences, covering all required topics.
- [x] `rag.py` provides fixed-overlap and sentence-based chunking, local SentenceTransformer embeddings, separate ChromaDB collections, and `collection.upsert()`.
- [x] Threshold calibration uses five in-scope queries and two out-of-scope queries; the main demonstration prints all five in-scope answers and both out-of-scope fallbacks.
- [x] Per-query document precision/recall arithmetic and a quantified recommendation are printed for both chunking strategies.

Commands:
```powershell
python dataset.py
python rag.py
```

Offline prerequisite: cache `sentence-transformers/all-MiniLM-L6-v2` locally before grading. `rag.py` sets Hugging Face and Transformers offline flags so graded RAG runs do not download a model.

## Part 2 — CrewAI workflow, memory, schema, and guardrails

- [x] `crew_support.py` defines three agents and three sequential tasks.
- [x] Policy retrieval and ticket lookup tools are assigned to their specialized agents; the composer has no tools. The demo prints per-agent assignments and tool execution logs.
- [x] CrewAI kickoff results are validated through the Pydantic `SupportResponse` schema.
- [x] Ticket lookup returns status, resolution time, and a score in [0, 1] using `0.60 * escalated + 0.40 * (1 - days_since_created / 30)`. Its review cutoff is the nearest-rank 80th percentile of the generated dataset's score distribution.
- [x] LangChain message history supports ticket-reference carryover, session isolation, and reset; the control transcript shows a follow-up query and the fresh/reset-session fallback.
- [x] Input phone masking, prompt-injection blocking, and output grounding checks have regression tests.
- [x] All graded examples use deterministic mock clients. CrewAI telemetry-disable flags are set before CrewAI initialization.

Commands:
```powershell
python -m unittest test_ticket_lookup.py -v
python -m unittest test_output_grounding.py -v
python test_guardrails.py
python crew_support.py
python run_capstone_evidence.py
```

## Part 3 — API, logs, and evaluation

- [x] FastAPI provides `GET /`, `POST /ask`, `POST /add-document`, and `WS /ws` with Pydantic request models.
- [x] HTTP middleware logs one JSONL metadata event per HTTP request, including trace ID, endpoint, status, and duration; request content and phone numbers are not logged.
- [x] WebSocket messages have trace/timing logs and the handler catches clean client disconnects.
- [x] The support evaluator covers 12 knowledge-base topics, an out-of-scope query, a missing ticket, and prompt injection, scoring Accuracy, Grounding, Completeness, and Safety per case and on average.
- [x] Evaluation scores are explicitly labeled rule-based heuristics, not independent quality judgments.

Commands:
```powershell
python -m unittest test_api_logging.py -v
python evaluate_support.py
python test_websocket.py
```

For `test_websocket.py`, start the API in another terminal first. The RAG model must be cached for policy-query tests.

## Part 4 — AutoGen review, governance, budgets, and cache

- [x] `autogen_review.py` runs a two-agent `RoundRobinGroupChat(max_turns=2)` with `Final-Editor` structured output via `VerdictModel(approved, final_answer, reason)` and `StructuredMessage[VerdictModel]`.
- [x] Evidence demonstrates a grounded answer approved unchanged and an intentionally unsupported 10-minute guarantee removed in a real revision.
- [x] Governance classifies LOW, MEDIUM, and HIGH risk with rationale and restricts consequential actions to human review or blocking.
- [x] Per-request approximate token and synthetic budget limits reject oversized/over-budget input before tools. The synthetic cost rate is for testing controls only; `MOCK_LLM` incurs no provider charges.
- [x] Cache keys use normalized query text. Evidence prints elapsed milliseconds and before/after counters showing the repeated query is a cache hit and the ticket tool is not called again.

Commands:
```powershell
python -m unittest test_autogen_review.py -v
python -m unittest test_workflow_controls.py -v
python autogen_review.py
python run_capstone_evidence.py
```

## Checked-in evidence

- `evidence/autogen_review_transcript.txt` — actual deterministic CI transcript for approval and revision.
- `evidence/capstone_controls_transcript.txt` — actual deterministic CI transcript for cache, memory, token limit, and budget controls.
- The GitHub Actions workflow compiles the Python files, runs the offline regression suite, regenerates both transcripts, and uploads them as the `capstone-text-evidence` artifact.

## Important scope limitations

- The policy documents and tickets are illustrative synthetic data, not official Ola policy or live system records.
- The 15-case support evaluation uses deterministic rule-based heuristics; a score of 1.0 is not proof of perfect answer quality.
- Governance marks requests for review but does not connect to a human approval queue.
- The mock budget rate is an internal demonstration value, not an external provider price.
- The CI smoke suite intentionally avoids running RAG/CrewAI policy retrieval because it requires the locally cached embedding model; run the commands above in the prepared local environment before submission.
