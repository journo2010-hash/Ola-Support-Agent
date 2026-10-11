
# Ola Domain Support Agent — CrewAI

A capstone project demonstrating an AI-assisted customer-support workflow using a synthetic ticket dataset, a Markdown knowledge base, retrieval-augmented generation (RAG), CrewAI agents, session memory, guardrails, governance controls, a FastAPI interface, and an AutoGen review workflow.

> **Important:** The knowledge-base policies and ticket records are illustrative test data. They are not official Ola policies or real customer records.

## Features

- **Synthetic dataset:** Reproducible support tickets across five categories and five statuses.
- **Knowledge base:** 12 Markdown policy documents covering priority, SLAs, escalation, refunds, communication channels, business hours, repeat complaints, service credits, feedback, VIP handling, outages, and data retention.
- **RAG:** Fixed-overlap and sentence-based chunking, local SentenceTransformer embeddings, ChromaDB collections, and hybrid semantic/lexical retrieval.
- **Out-of-scope handling:** Similarity threshold calibration and a fallback when the knowledge base does not provide sufficient evidence.
- **CrewAI:** Three agents and three sequential tasks for policy retrieval, ticket lookup, and response composition.
- **Ticket lookup:** Synthetic ticket status, resolution time, and escalation scoring.
- **Session memory:** Multi-turn conversation history and session reset.
- **Guardrails:** Phone-number masking, prompt-injection detection, and input-length limits.
- **Governance:** Rule-based risk classification, autonomy decisions, and human-review requirements for sensitive requests.
- **Response cache:** In-memory response caching with a time-to-live and normalized query keys.
- **Structured logging:** JSONL request logs designed to avoid recording raw question text or unmasked phone numbers.
- **FastAPI:** HTTP endpoints and WebSocket support.
- **Evaluation:** A 15-case support evaluation and CSV output, plus a separate RAG chunking comparison.
- **AutoGen review:** Two-agent review demonstration with structured Pydantic verdicts.

## Project files

| File or folder | Purpose |
|---|---|
| `dataset.py` | Generates and validates synthetic support tickets |
| `rag.py` | Knowledge-base loading, chunking, embeddings, retrieval, and threshold calibration |
| `rag_tfidf.py` | Alternative TF-IDF retrieval implementation |
| `ticket_lookup.py` | Ticket status lookup and dataset-calibrated escalation scoring |
| `run_capstone_evidence.py` | Reproducible cache, session-memory, and request-budget transcript |
| `test_workflow_controls.py` | Cache, session-context, token-limit, and budget regression tests |
| `test_autogen_review.py` | Structured AutoGen approval/revision regression tests |
| `test_workflow_controls.py` | Cache, session-context, token-limit, and budget regression tests |
| `test_ticket_lookup.py` | Escalation-score formula and percentile-threshold tests |
| `test_output_grounding.py` | Output-grounding guard regression tests |
| `test_api_logging.py` | One-log-per-request, trace ID, and PII-safe API logging tests |
| `run_capstone_evidence.py` | Reproducible cache, memory, and budget transcript |
| `requirements-ci.txt` | Minimal offline test dependencies |
| `.github/workflows/capstone-smoke.yml` | Deterministic smoke-test workflow |
| `CAPSTONE_RUBRIC_CHECKLIST.md` | Rubric-to-implementation and evidence checklist |
| `CAPSTONE_DEMO_GUIDE.md` | Five-minute demo script and evaluator Q&A |
| `evidence/autogen_review_transcript.txt` | Actual CI output for unchanged approval and grounded revision |
| `evidence/capstone_controls_transcript.txt` | Actual CI output for cache, session carryover/reset, and budget controls |
| `crew_support.py` | CrewAI agents, tasks, and tools |
| `mock_llm.py` | Deterministic mock language model for local testing |
| `mock_workflow.py` | Support workflow, memory, guardrails, caching, and governance integration |
| `session_memory.py` | Session conversation history |
| `guardrails.py` | Input validation, prompt-injection detection, and phone masking |
| `governance.py` | Rule-based risk classification and autonomy decisions |
| `response_models.py` | Structured response validation |
| `support_agents.py` | Support-agent definitions |
| `api.py` | FastAPI application, HTTP endpoints, WebSocket, and JSONL logging |
| `evaluate_support.py` | Runs the 15-case support evaluation |
| `evaluation_results.csv` | Generated support evaluation results |
| `autogen_review.py` | AutoGen review demonstration |
| `test_guardrails.py` | Guardrail regression tests |
| `test_governance.py` | Governance unit tests |
| `test_websocket.py` | WebSocket request and disconnection test |
| `knowledge_base/` | Markdown support policy documents |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.11
- Git
- A virtual environment
- Dependencies listed in `requirements.txt`

The project uses CrewAI, ChromaDB, SentenceTransformers, LangChain, FastAPI, Pydantic, AutoGen, and related packages.

Before graded CrewAI runs, disable telemetry explicitly in PowerShell:

```powershell
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:OTEL_SDK_DISABLED = "true"
```

The CrewAI and AutoGen demo scripts also set these environment variables before initializing their agent frameworks.

## Installation on Windows

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/journo2010-hash/Ola-Support-Agent.git
cd Ola-Support-Agent
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check dependency consistency:

```powershell
python -m pip check
```

These commands describe a fresh installation. If the project and its virtual environment already exist, continue using that environment.

## Dataset design and reproducibility

The default dataset contains 50 synthetic support tickets. The generator uses Python's seeded random generator and starts with a base seed of `42`.

- **Categories:** Billing, Technical Issue, Account Access, Product Defect, and General Inquiry.
- **Category weights/distribution:** Equal category weighting (20% each); 10 tickets per category in the default 50-ticket dataset.
- **Statuses:** Open, In Progress, Escalated, Resolved, and Closed.
- **Status assignment weights:** A ticket is Escalated exactly when its Boolean escalation flag is true. Non-escalated tickets are distributed evenly and cyclically across Open, In Progress, Resolved, and Closed (25% each among non-escalated tickets).
- **Escalation flag weight:** 20% true / 80% false per random draw during generation.
- **Escalation acceptance range:** 10–30% of tickets; the generator retries subsequent seeds until the generated sample is inside this range.
- **Ticket age:** 0–30 days.
- **Resolution time:** 0.5–120 hours.
- **Record identifiers:** Generated IDs such as `OLA-0001`.
- **Consistency rule:** The `Escalated` status must agree with the Boolean `escalated` field.

The generator retries subsequent seeds if the generated escalation percentage falls outside the required range. Therefore, the actual seed can differ from the base seed.

The validated default run used seed `42` and produced:

- 50 tickets.
- 10 tickets in each category.
- All five required statuses.
- 6 escalated tickets, corresponding to a 12% escalation rate; the other four statuses each contain 11 tickets in this run.
- Ticket ages within 0–30 days.
- Resolution times within 0.5–120 hours.

Run the dataset generator and validation:

```powershell
python dataset.py
```

The data is synthetic and is intended for reproducible demonstrations and testing, not as a representation of real customer-support statistics.

## Knowledge base

The `knowledge_base/` directory contains 12 Markdown policy documents:

1. `ticket_priority.md`
2. `sla_policy.md`
3. `escalation_matrix.md`
4. `refund_policy.md`
5. `communication_channels.md`
6. `business_hours.md`
7. `repeat_complaints.md`
8. `service_credit.md`
9. `feedback_collection.md`
10. `vip_handling.md`
11. `outage_communication.md`
12. `data_retention.md`

These documents provide the context for support-policy questions. Answers should be grounded in retrieved policy content rather than unsupported assumptions.

## RAG implementation and evaluation

The RAG implementation compares two document chunking strategies:

- **Fixed-overlap chunking:** Divides documents into overlapping text chunks, using 80-word chunks and 15-word overlap by default.
- **Sentence-based chunking:** Splits documents into sentence-oriented chunks.

Both approaches use local SentenceTransformer embeddings and separate ChromaDB collections. The implementation uses `collection.upsert()` to index generated chunks.

For graded runs, `rag.py` forces Hugging Face/Transformers offline mode. Ensure `sentence-transformers/all-MiniLM-L6-v2` is already present in the local model cache before running RAG, CrewAI policy retrieval, or the full support evaluation; the application will not download it at runtime.

Hybrid retrieval combines:

- 75% semantic similarity.
- 25% lexical keyword overlap.

A similarity threshold is calibrated using in-scope and out-of-scope examples. The system returns an uncertainty fallback when the retrieved evidence does not meet the configured threshold.

Run the RAG demonstration:

```powershell
python rag.py
```

### Preliminary chunking comparison

The current five-query document-level evaluation produced the following results:

| Metric | Fixed-overlap | Sentence-based |
|---|---:|---:|
| Mean document-level precision | 1.000 | 1.000 |
| Mean document-level recall | 1.000 | 1.000 |

Both strategies retrieved the expected document for all five tested queries, giving each a mean document-level precision of `1.000` and recall of `1.000`. I recommend keeping sentence-based chunking as the default for its fine-grained source units while retaining fixed-overlap as a comparison; this is an implementation preference, not evidence that it is more accurate. Expand the query set before claiming a performance advantage.

The threshold calibration tested five in-scope queries and two out-of-scope questions. The selected thresholds separated the observed scores in this small sample.

The two tested out-of-scope questions, about sourdough bread and the capital of Japan, triggered the fallback:

> I don't know based on the available support policies.

These results are preliminary. A larger and more varied query set is needed to assess retrieval quality and fallback reliability more broadly.

## CrewAI support workflow

The CrewAI workflow defines three agents:

- **Support Policy Retrieval Specialist:** Retrieves policy evidence using the RAG tool.
- **Support Ticket Lookup Specialist:** Looks up synthetic ticket information using the ticket-status tool.
- **Support Response Composer:** Composes a response using preceding task outputs.

The workflow uses three sequential tasks. The composer task receives the retrieval and lookup task outputs as context.

The local mock language model supports deterministic testing without requiring a live language-model API for the demonstrated workflow.

A successful local kickoff test recorded calls to both `search_support_policies` and `check_support_ticket_status`. For ticket `OLA-0001`, the sample result reported:

- Status: `Open`
- Category: `Billing`
- Resolution time: `91.62` hours
- Escalation score: `0.173`

Ticket details are synthetic and are included only to demonstrate tool integration.

Run the CrewAI demonstration:

```powershell
python crew_support.py
```

The script displays the CrewAI result, the recorded tool-execution log, and a separate deterministic offline tool-verification result.

## Ticket lookup and escalation scoring

The `ticket_lookup.py` module exposes:

```python
check_support_ticket_status(record_id)
```

The function returns ticket status, resolution time, and an escalation score between 0 and 1. The score combines escalation status with normalized ticket recency.

The score is a transparent triage heuristic, not a calibrated probability:
`score = 0.60 × escalated_boolean + 0.40 × (1 − days_since_created / 30)`.
The implementation computes a nearest-rank **80th-percentile threshold from the generated dataset's own score distribution** and returns `priority_review_flag` when the score meets that threshold. Ties can cause slightly more than 20% of tickets to be flagged; the score must not be described as a probability.

## Session memory and structured responses

The session-memory implementation uses LangChain message-history components to maintain conversation history across turns.

The workflow stores messages in LangChain's `InMemoryChatMessageHistory`. When a follow-up asks about “its” status, score, or resolution, the workflow resolves a ticket ID only from that same session's recent history. A fresh session has no such context, and resetting a session removes its conversation history.

Run the deterministic carryover/reset and cache/budget evidence demonstration:

```powershell
python run_capstone_evidence.py
```

The script writes its actual console output to `evidence/capstone_controls_transcript.txt`. The checked-in transcript was captured from a successful deterministic CI run; the CI workflow also uploads both transcript files as the `capstone-text-evidence` artifact. It demonstrates a ticket lookup followed by an escalation-score follow-up, then shows that a fresh or reset session cannot resolve the pronoun without a ticket ID. It also prints counters before and after a normalized-query cache hit and verifies that the second identical query does not call the ticket tool again.

Pydantic models validate structured support responses and AutoGen review verdicts. Session history is in memory and is not durable production storage.

## Guardrails and privacy

The guardrail implementation demonstrates:

- Masking fixed-format phone numbers.
- Detecting prompt-injection attempts.
- Rejecting or limiting oversized inputs.
- Returning safe refusals for blocked requests.
- Avoiding support-tool calls for rejected prompt-injection inputs.

Run the guardrail regression tests:

```powershell
python test_guardrails.py
```

The latest recorded run passed all four checks.

Phone masking and prompt-injection detection are safeguards, not guarantees against every unsafe input. Do not use real customer data in the demonstration.

## Governance controls

The `governance.py` module classifies requests by risk and assigns an autonomy level. It distinguishes read-only support requests from sensitive actions that require human review and requests that must be blocked.

The implemented test cases cover:

- Low-risk support-policy questions.
- Read-only ticket lookups.
- Sensitive actions, including refund requests and account changes.
- Prompt-injection attempts.
- Empty or invalid requests.
- Serialization of governance decisions.

Run the governance unit tests:

```powershell
python -m unittest test_governance.py -v
```

The latest recorded run passed all nine governance unit tests.

**Limitations:** The governance implementation uses rule-based classification. Flagging a request for human review does not implement an actual human approval workflow. The rules do not guarantee detection of every unsafe request.

## FastAPI and WebSocket interface

Start the development server:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

The application includes:

- `GET /` — basic service information.
- `POST /ask` — submits a support question.
- `POST /add-document` — adds a document through the API.
- `WS /ws` — provides a WebSocket support interaction.

The WebSocket implementation handles normal client disconnection using `WebSocketDisconnect` and offloads workflow execution to avoid blocking the asynchronous connection.

The CrewAI entry point validates each `.kickoff()` result through `SupportResponse` from `response_models.py`. It runs separate policy and ticket queries and prints the actual tool-execution log for each query. It also prints per-agent tool assignments so the least-autonomy design is visible: only the ticket lookup specialist receives the ticket lookup tool; the composer receives no tools.

### WebSocket test

Keep the API server running in one terminal. Open a second PowerShell terminal in the project directory, activate the virtual environment, and run:

```powershell
python test_websocket.py
```

The recorded test passed and returned a policy response with governance metadata. It also confirmed clean client disconnection.

Stop the API server with `Ctrl + C`.

## Structured logging and response caching

The API writes structured JSONL request logs containing trace identifiers, timing, endpoint information, outcomes, and status codes.

The implementation is designed not to record raw question text or unmasked phone numbers. Review generated logs before sharing or committing them.

The local response cache uses **normalized query text** as its key, independent of session ID, and has a time-to-live. `get_workflow_metrics()` exposes cache hits/misses and tool-call counters so the cache behavior can be verified empirically. It is not a distributed or persistent cache.

The request guard also enforces an approximate input-token limit and an illustrative per-request budget in `MOCK_LLM` mode. The synthetic budget rate is only for exercising the control path; it is **not a real model-provider price**, and the mock workflow makes no billable model calls.

## Support evaluation

Run the support evaluation:

```powershell
python evaluate_support.py
```

The script writes results to:

```text
evaluation_results.csv
```

The 15-case evaluation includes:

- One representative query for each of the 12 knowledge-base topics.
- An out-of-scope question about sourdough bread.
- A missing-ticket query.
- A prompt-injection guardrail case.

The evaluation reports accuracy, grounding, completeness, and safety scores per query and averages.

A previous recorded run returned an average score of `1.000` for all four dimensions, and all 15 cases scored `1.0` on each of the four heuristics. Re-run the evaluation after any code change and use the newly generated CSV as the submission evidence.

**Interpretation limitation:** These scores are based on rule-based heuristic checks, not independent human review or model-independent measurements. They should not be interpreted as proof of perfect answer quality.

Review the CSV, retrieved documents, final responses, and edge cases before presenting the results.

## AutoGen review workflow

Run the AutoGen demonstration:

```powershell
python autogen_review.py
```

The workflow uses two agents in a `RoundRobinGroupChat` with a maximum of two agent turns and a structured Pydantic `Verdict`.

It demonstrates two scenarios:

- **APPROVE:** The response includes a supporting source.
- **REVISE:** The response is missing a supporting source.

Each run asserts both outcomes: the grounded draft must be approved unchanged, and the deliberately ungrounded draft must be revised.

The structured schema is `VerdictModel(approved: bool, final_answer: str, reason: str)`; the `Final-Editor` uses `output_content_type=VerdictModel`, and the team registers `StructuredMessage[VerdictModel]` with `max_turns=2`. The approval scenario asserts the final answer is unchanged. The revision scenario injects an unsupported 10-minute guarantee and asserts the final answer removes that claim.

Run the focused regression tests:

```powershell
python -m unittest test_autogen_review.py -v
python -m unittest test_workflow_controls.py -v
python -m unittest test_ticket_lookup.py -v
python -m unittest test_output_grounding.py -v
python -m unittest test_api_logging.py -v
```

Running `python autogen_review.py` regenerates `evidence/autogen_review_transcript.txt`. The checked-in transcript was captured from a successful deterministic CI run; the CI workflow also uploads the generated transcript files as the `capstone-text-evidence` artifact.

This is a deterministic mock demonstration, not an evaluation of a production language model.

## Dependency and regression checks

Check installed dependency consistency:

```powershell
python -m pip check
```

The previously recorded run returned:

```text
No broken requirements found.
```

Run the main checks individually:

```powershell
python dataset.py
python -m unittest test_governance.py -v
python test_guardrails.py
python -m unittest test_workflow_controls.py -v
python -m unittest test_ticket_lookup.py -v
python -m unittest test_output_grounding.py -v
python -m unittest test_api_logging.py -v
python -m unittest test_autogen_review.py -v
python run_capstone_evidence.py
python rag.py
python evaluate_support.py
python crew_support.py
python autogen_review.py
```

The WebSocket test requires the API server to be running in a separate terminal:

```powershell
python test_websocket.py
```

## Safety, governance, and deployment limitations

This repository is a local capstone demonstration.

- Use synthetic tickets for demonstrations.
- Do not commit API keys, access tokens, phone numbers, or other secrets.
- Review `.gitignore` before committing generated logs, virtual environments, or sensitive files.
- Apply input-length and per-request resource limits.
- Treat retrieved policy documents as evidence for support-policy answers.
- Use safe fallbacks when the available knowledge base does not support an answer.
- Apply authentication, authorization, rate limits, secure storage, retention controls, and a security review before any real deployment.
- Evaluate the system using a larger, diverse query set before making reliability claims.

The mock language model, synthetic tickets, local cache, and in-memory session history are for testing and demonstration. They do not establish production readiness.

## Repository

GitHub: https://github.com/journo2010-hash/Ola-Support-Agent

## Reproducibility notes

Record the following when reproducing results:

- Python version and dependency versions from `requirements.txt`.
- Dataset base seed and actual seed used.
- Dataset generation weights and acceptance rules.
- RAG chunking strategy and calibrated threshold.
- Evaluation query set, per-query results, and aggregation method.
- Whether a result came from a deterministic mock workflow or a live model.

Keep reported results tied to the exact test run, and distinguish measured behavior from assumptions or intended behavior.
