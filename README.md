
# Ola Domain Support Agent - CrewAI

A capstone project demonstrating an AI-assisted customer-support workflow using a synthetic ticket dataset, a knowledge base, retrieval-augmented generation (RAG), CrewAI agents, session memory, guardrails, a FastAPI interface, and an AutoGen review workflow.

> **Important:** The knowledge-base policies and ticket records are illustrative test data. They are not official Ola policies or real customer records.

## Features

- **Synthetic dataset:** Seeded support tickets across Billing, Technical Issue, Account Access, Product Defect, and General Inquiry.
- **Knowledge base:** Markdown policy documents covering priority, SLAs, escalation, refunds, communication, retention, and other support topics.
- **RAG:** Document chunking, local embeddings, ChromaDB retrieval, and a similarity threshold for out-of-scope questions.
- **CrewAI:** Retrieval, ticket lookup, and response-composition agents.
- **Ticket lookup:** Status and escalation scoring for synthetic ticket IDs.
- **Session memory:** Multi-turn conversation history and session reset.
- **Guardrails:** Phone-number masking, prompt-injection detection, and input-length limits.
- **Response cache:** In-memory cache with a time-to-live.
- **Structured logging:** JSONL request logs designed to avoid recording raw question text or phone numbers.
- **FastAPI:** HTTP endpoints and WebSocket support.
- **Evaluation:** A 15-query evaluation script and CSV output.
- **AutoGen review:** Two-agent review with structured Pydantic verdicts.

## Project files

| File | Purpose |
|---|---|
| `dataset.py` | Generates and validates synthetic tickets |
| `rag.py` | Knowledge-base loading, chunking, retrieval, and threshold logic |
| `ticket_lookup.py` | Looks up synthetic ticket status and escalation score |
| `crew_support.py` | CrewAI agent and task definitions |
| `mock_workflow.py` | Local mock workflow, memory, guardrails, and caching |
| `session_memory.py` | Session conversation history |
| `guardrails.py` | Input checks and phone-number masking |
| `response_models.py` | Structured response validation |
| `api.py` | FastAPI application and JSONL request logging |
| `evaluate_support.py` | Runs the preliminary 15-query evaluation |
| `evaluation_results.csv` | Saved evaluation output |
| `autogen_review.py` | Local AutoGen approval/revision demonstration |
| `knowledge_base/` | Markdown support policy documents |
| `test_guardrails.py` | Guardrail regression tests |

## Requirements

- Python 3.11
- A virtual environment
- Dependencies used by the project, including CrewAI, LangChain, SentenceTransformers, ChromaDB, FastAPI, Pydantic, and AutoGen

Install dependencies from a project requirements file if one is provided. If the repository does not yet contain a `requirements.txt`, create one from the tested environment before setting up the project on another computer.

## Run locally on Windows

Open PowerShell in the repository folder.

### 1. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run guardrail tests

```powershell
.\.venv\Scripts\python.exe test_guardrails.py
```

### 3. Run the evaluation

```powershell
.\.venv\Scripts\python.exe evaluate_support.py
```

The script writes evaluation results to `evaluation_results.csv`.

The current evaluation uses rule-based scoring. Its reported scores are preliminary indicators, not independent verification of answer quality. Review answers, sources, and edge cases before presenting these as final performance metrics.

### 4. Run the AutoGen review demo

```powershell
.\.venv\Scripts\python.exe autogen_review.py
```

The deterministic mock client demonstrates two scenarios:

- **APPROVE:** The answer includes a supporting source.
- **REVISE:** The answer is missing a supporting source.

The team uses two AutoGen agents and a structured `Verdict` model. The initial task plus two agent replies result in three messages, representing two agent turns.

This is a deterministic mock demonstration, not a production language model. It does not measure the quality of a real LLM.

### 5. Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

Stop the development server with `Ctrl + C`.

## Evaluation methodology

The evaluation script uses 15 representative queries: one for each of the 12 knowledge-base topics and three additional out-of-scope or edge-case checks.

The output includes accuracy, grounding, completeness, and safety scores. These are heuristic checks, not human-reviewed or model-independent quality measurements. Inspect the CSV and source documents before reporting performance claims.

For RAG chunking comparison, document-level precision and recall should be calculated on the same query set for both chunking strategies. Document the per-query results, aggregation method, and recommendation.

## Privacy and safety

- Use synthetic tickets only in demonstrations.
- Do not put API keys, access tokens, phone numbers, or other secrets in Git.
- Review `.gitignore` before committing generated logs, virtual environments, or sensitive files.
- Phone masking and prompt-injection detection are guardrails, not guarantees against every unsafe input.
- The in-memory cache and session history are local application features, not durable production storage.
- Apply authentication, authorization, retention controls, rate limits, and security review before any real deployment.

## Reproducibility

Document the dataset seed, generation weights, ticket amount range, and dependency versions alongside evaluation results. Keep test outputs reproducible and distinguish synthetic/mock results from live-system results.

## Repository

GitHub: https://github.com/journo2010-hash/Ola-Support-Agent
