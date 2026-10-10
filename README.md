# Ola Domain Support Agent — CrewAI

A capstone project demonstrating an AI-assisted customer-support workflow using a synthetic ticket dataset, a Markdown knowledge base, retrieval-augmented generation (RAG), CrewAI agents, session memory, guardrails, a FastAPI interface, and an AutoGen review workflow.

> **Important:** The knowledge-base policies and ticket records are illustrative test data. They are not official Ola policies or real customer records.

## Features

- **Synthetic dataset:** Reproducible support tickets across five categories and five statuses.
- **Knowledge base:** 12 Markdown policy documents covering ticket priority, SLAs, escalation, refunds, communication channels, business hours, repeat complaints, service credits, feedback, VIP handling, outages, and data retention.
- **RAG:** Fixed-overlap and sentence-based chunking, local SentenceTransformer embeddings, ChromaDB collections, and hybrid semantic/lexical retrieval.
- **Out-of-scope handling:** Empirically calibrated similarity thresholds and a fallback when the available support policies do not provide sufficient evidence.
- **CrewAI:** Retrieval, ticket lookup, and response-composition agents.
- **Ticket lookup:** Synthetic ticket status, resolution time, and escalation scoring.
- **Session memory:** Multi-turn conversation history and session reset.
- **Guardrails:** Phone-number masking, prompt-injection detection, and input-length limits.
- **Response cache:** In-memory response caching with a time-to-live and normalized query keys.
- **Structured logging:** JSONL request logs with trace IDs and timing information, designed to avoid logging raw question text or phone numbers.
- **FastAPI:** HTTP endpoints and WebSocket support.
- **Evaluation:** A 15-query support evaluation and CSV output, plus a separate RAG chunking comparison.
- **AutoGen review:** Two-agent review with structured Pydantic verdicts.

## Project files

| File or folder | Purpose |
|---|---|
| `dataset.py` | Generates and validates synthetic support tickets |
| `rag.py` | Knowledge-base loading, chunking, embeddings, retrieval, and threshold calibration |
| `rag_tfidf.py` | Alternative TF-IDF retrieval implementation |
| `ticket_lookup.py` | Looks up ticket status and calculates escalation scores |
| `crew_support.py` | CrewAI agent and task definitions |
| `mock_llm.py` | Deterministic mock language model for local testing |
| `mock_workflow.py` | Local support workflow, memory, guardrails, and caching |
| `session_memory.py` | Session conversation history |
| `guardrails.py` | Input validation, prompt-injection detection, and phone masking |
| `response_models.py` | Structured response validation |
| `api.py` | FastAPI application, HTTP endpoints, WebSocket, and JSONL logging |
| `evaluate_support.py` | Runs the 15-query support evaluation |
| `evaluation_results.csv` | Saved support evaluation results |
| `autogen_review.py` | AutoGen approval/revision demonstration |
| `test_guardrails.py` | Guardrail regression tests |
| `test_websocket.py` | WebSocket request and disconnection test |
| `knowledge_base/` | Markdown support policy documents |
| `requirements.txt` | Python dependency versions |

## Requirements

- Python 3.11
- Git
- A virtual environment
- The dependencies listed in `requirements.txt`

The project uses CrewAI, ChromaDB, SentenceTransformers, LangChain, FastAPI, Pydantic, AutoGen, and related packages.

## Installation on Windows

Clone the repository and enter its directory:

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

Check for dependency conflicts:

```powershell
python -m pip check
```

The commands above describe a fresh installation. If you already have the project and its `.venv` environment, you can continue using that environment.

## Dataset design and reproducibility

The default dataset contains 50 synthetic support tickets. The generator uses Python's seeded random generator and starts with a base seed of `42`.

- **Categories:** Billing, Technical Issue, Account Access, Product Defect, and General Inquiry.
- **Category distribution:** 10 tickets per category in the default dataset.
- **Statuses:** Open, In Progress, Escalated, Resolved, and Closed.
- **Escalation weights:** 20% escalated and 80% non-escalated during generation.
- **Escalation acceptance range:** 10–30% of tickets.
- **Ticket age:** 0–30 days since creation.
- **Resolution time:** 0.5–120 hours.
- **Record identifiers:** Generated IDs such as `OLA-0001`.
- **Consistency rule:** The `Escalated` status must agree with the Boolean `escalated` field.

The generator retries subsequent seeds if the generated escalation percentage falls outside the required range. Therefore, the actual seed can differ from the base seed.

The validated default run used seed `42` and produced:

- 50 tickets.
- 10 tickets in each category.
- All five required statuses.
- 6 escalated tickets, corresponding to a 12% escalation rate.
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

- **Fixed-overlap chunking:** Divides documents into chunks with overlapping text.
- **Sentence-based chunking:** Splits documents into sentence-oriented chunks.

Both approaches use local SentenceTransformer embeddings and separate ChromaDB collections. The implementation uses `collection.upsert()` to index the generated chunks.

Hybrid retrieval combines semantic similarity with lexical overlap. A similarity threshold is calibrated using in-scope and out-of-scope examples, and the system returns an uncertainty fallback when the query does not have sufficient support.

Run the RAG demonstration:

```powershell
python rag.py
```

### Preliminary chunking comparison

The latest five-query document-level evaluation produced the following results:

| Metric | Fixed-overlap | Sentence-based |
|---|---:|---:|
| Mean document-level precision | 1.000 | 1.000 |
| Mean document-level recall | 1.000 | 1.000 |

On this small evaluation set, both approaches retrieved the expected document for all five queries. The two strategies therefore tie on the reported metrics; these results do not establish that either approach is universally better.

The threshold calibration also tested five in-scope queries and two out-of-scope questions. The selected thresholds separated the tested in-scope and out-of-scope similarity scores.

Two tested out-of-scope questions, about sourdough bread and the capital of Japan, triggered the fallback:

> I don't know based on the available support policies.

These results are preliminary. A larger and more varied query set is needed to assess retrieval quality and fallback reliability more broadly.

## CrewAI support workflow

The CrewAI workflow defines three agents:

- **Retrieval Agent:** Searches the support knowledge base using the RAG tool.
- **Lookup Agent:** Checks synthetic ticket details using the ticket-status tool.
- **Composer Agent:** Produces a final support response from the available information.

The workflow uses CrewAI tasks and a sequential process. The local mock language model supports deterministic testing without requiring a live language-model API for the demonstrated workflow.

A successful local workflow test executed both the policy retrieval and ticket lookup tools. For ticket `OLA-0001`, the sample result reported status `Open`, category `Billing`, resolution time `91.62` hours, and escalation score `0.173`.

The ticket details are synthetic and are included only to demonstrate tool integration.

## Ticket lookup and escalation scoring

The `ticket_lookup.py` module exposes:

```python
check_support_ticket_status(record_id)
```

The function returns ticket status, resolution time, and an escalation score between 0 and 1. The score combines escalation status with normalized ticket recency.

The score is a demonstration metric, not a calibrated probability of escalation. Interpret it according to the formula implemented in the source code.

## Session memory and structured responses

The session-memory implementation uses LangChain message-history components to maintain conversation history across turns.

The demonstration covers:

- Adding user and assistant messages.
- Retrieving history for a session.
- Keeping different sessions isolated.
- Clearing a session to start fresh.

Pydantic models validate structured support responses and AutoGen review verdicts. Session history is maintained in memory and is not durable production storage.

## Guardrails and privacy

The guardrail implementation demonstrates:

- Masking fixed-format phone numbers.
- Detecting prompt-injection attempts.
- Rejecting or limiting oversized inputs.
- Returning safe refusals for blocked requests.
- Avoiding calls to support tools for rejected prompt-injection inputs.

Run the guardrail regression tests:

```powershell
python test_guardrails.py
```

The latest recorded guardrail regression test passed all four checks.

Phone masking and prompt-injection detection are safeguards, not guarantees against every unsafe input. Do not use real customer data in the demonstration.

## FastAPI and WebSocket interface

Start the development server:

```powershell
python -m uvicorn api:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

The application includes:

- `GET /` — basic service information.
- `POST /ask` — submits a support question.
- `POST /add-document` — adds a document through the API.
- `WS /ws` — provides a WebSocket support interaction.

The WebSocket implementation handles normal client disconnection using `WebSocketDisconnect`.

### WebSocket test

Keep the API server running in one terminal. Open a second PowerShell terminal in the project directory, activate the virtual environment, and run:

```powershell
python test_websocket.py
```

The recorded test completed successfully and confirmed clean client disconnection.

Stop the API server with `Ctrl + C`.

## Structured logging and response caching

The API writes structured JSONL request logs containing trace identifiers, timing, endpoint information, outcomes, and status codes.

The implementation is designed not to record raw question text, session identifiers, or unmasked phone numbers. Review generated logs before sharing or committing them.

The local response cache uses normalized query keys and a time-to-live. It demonstrates repeated-query caching but is not a distributed or persistent cache.

## Support evaluation

Run the support evaluation:

```powershell
python evaluate_support.py
```

The script writes results to:

```text
evaluation_results.csv
```

The 15-query evaluation includes:

- One representative query for each of the 12 knowledge-base topics.
- An out-of-scope question about sourdough bread.
- A missing-ticket query.
- A prompt-injection guardrail case.

The evaluation reports accuracy, grounding, completeness, and safety scores per query and averages.

The latest recorded run returned an average score of `1.000` for all four dimensions. These scores are based on rule-based heuristic checks, not independent human review or model-independent measurements. They should not be interpreted as proof of perfect answer quality.

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

The recorded scenario tests passed for both outcomes.

This is a deterministic mock demonstration, not an evaluation of a production language model.

## Dependency and regression checks

Check installed dependency consistency:

```powershell
python -m pip check
```

The latest recorded run returned:

```text
No broken requirements found.
```

Run the relevant tests individually:

```powershell
python dataset.py
python test_guardrails.py
python evaluate_support.py
python rag.py
python autogen_review.py
```

The WebSocket test requires the API server to be running in a separate terminal.

## Safety, governance, and deployment limitations

This repository is a local capstone demonstration.

- Use synthetic tickets for demonstrations.
- Do not commit API keys, access tokens, phone numbers, or other secrets.
- Review `.gitignore` before committing generated logs, virtual environments, or sensitive files.
- Apply input-length and per-request resource limits.
- Treat retrieved policy documents as the evidence for support-policy answers.
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