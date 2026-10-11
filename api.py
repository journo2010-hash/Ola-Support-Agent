
"""FastAPI interface for the Ola Support Agent with safe JSONL logging."""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from mock_workflow import run_support_workflow


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "support_api.jsonl"

LOG_DIR.mkdir(parents=True, exist_ok=True)


class JsonlFormatter(logging.Formatter):
    """Format each log event as one JSON object per line."""

    def format(self, record):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        for key in (
            "trace_id",
            "endpoint",
            "duration_ms",
            "outcome",
            "status_code",
        ):
            value = getattr(record, key, None)
            if value is not None:
                event[key] = value

        return json.dumps(event, ensure_ascii=False)


logger = logging.getLogger("ola_support_api")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(JsonlFormatter())
    logger.addHandler(file_handler)


def log_request(trace_id, endpoint, started, outcome, status_code):
    """Log metadata only; never include request content or personal data."""

    logger.info(
        "request_completed",
        extra={
            "trace_id": trace_id,
            "endpoint": endpoint,
            "duration_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
            "outcome": outcome,
            "status_code": status_code,
        },
    )


app = FastAPI(
    title="Ola Support Agent API",
    description="Local mock support workflow with RAG and ticket lookup.",
    version="1.0.0",
)



@app.middleware("http")
async def log_every_http_request(request, call_next):
    """Log exactly one metadata-only JSONL event for every HTTP request."""
    trace_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log_request(trace_id, request.url.path, started, "error", 500)
        raise

    outcome = "success" if response.status_code < 400 else "error"
    log_request(trace_id, request.url.path, started, outcome, response.status_code)
    response.headers["X-Trace-ID"] = trace_id
    return response


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=100)



class CrewAskResponse(BaseModel):
    """Validated response from the CrewAI plus AutoGen endpoint."""

    answer: str
    mode: str = "MOCK_LLM"
    fallback: bool = False
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    ticket: dict | None = None
    question: str
    governance: dict
    review: dict | None = None


class AddDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=10, max_length=10000)


@app.get("/")
def root():
    """Basic health endpoint."""

    return {
        "service": "Ola Support Agent",
        "status": "running",
        "mode": "MOCK_LLM",
    }


@app.post("/ask")
def ask_support(request: AskRequest):
    """Answer a support question; HTTP middleware records safe metadata."""
    try:
        return run_support_workflow(
            request.question,
            session_id=request.session_id,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The support request could not be processed.",
        )


@app.post("/ask-crew", response_model=CrewAskResponse)
def ask_support_with_crew(request: AskRequest):
    """Run the actual CrewAI crew, then review its draft with AutoGen."""
    from governance import classify_request
    from guardrails import check_input, enforce_output_grounding, mask_phone_numbers
    from mock_workflow import (
        MAX_INPUT_TOKENS,
        MAX_REQUEST_COST_USD,
        estimate_mock_cost_usd,
        estimate_tokens,
    )

    original_question = request.question.strip()
    input_tokens = estimate_tokens(original_question)
    estimated_cost = estimate_mock_cost_usd(input_tokens)

    if input_tokens > MAX_INPUT_TOKENS or estimated_cost > MAX_REQUEST_COST_USD:
        return {
            "answer": "Your request exceeds the configured request limit. Please shorten it and try again.",
            "mode": "MOCK_LLM",
            "fallback": True,
            "sources": [],
            "tools_used": [],
            "question": mask_phone_numbers(original_question),
            "governance": {
                "risk_level": "high",
                "autonomy_level": "blocked",
                "allowed": False,
                "reason": "The configured input-token or mock-budget limit was exceeded.",
                "estimated_input_tokens": input_tokens,
                "estimated_mock_cost_usd": round(estimated_cost, 6),
                "max_request_cost_usd": MAX_REQUEST_COST_USD,
            },
        }

    guardrail = check_input(original_question)
    safe_question = guardrail.masked_text
    decision = classify_request(
        safe_question,
        guardrail_blocked=guardrail.blocked,
    )
    if guardrail.blocked or not decision.allowed:
        return {
            "answer": (
                "I can't process this request because it is blocked by the "
                "support agent's safety and governance rules."
            ),
            "mode": "MOCK_LLM",
            "fallback": True,
            "sources": [],
            "tools_used": [],
            "question": safe_question,
            "governance": decision.to_dict(),
        }

    try:
        from crew_support import (
            TOOL_EXECUTION_LOG,
            TOOL_RESULT_LOG,
            build_support_crew,
            validate_crew_response,
        )
        from autogen_review import review_draft

        TOOL_EXECUTION_LOG.clear()
        TOOL_RESULT_LOG.clear()
        crew = build_support_crew()
        raw_result = crew.kickoff(inputs={"question": safe_question})
        response = validate_crew_response(raw_result)

        policy_result = TOOL_RESULT_LOG.get("policy", {})
        ticket_result = TOOL_RESULT_LOG.get("ticket", {})
        retrieved_context = str(policy_result.get("answer", "")).strip()
        if ticket_result.get("found"):
            retrieved_context += (
                "\\nTicket lookup result: "
                + json.dumps(ticket_result, ensure_ascii=False)
            )

        grounding = enforce_output_grounding(
            answer=response.answer,
            sources=response.sources,
            fallback=response.fallback,
        )
        if grounding["fallback"] and not response.ticket:
            final_answer = grounding["answer"]
            review = {
                "approved": False,
                "reason": "The output grounding check rejected an answer without source evidence.",
            }
        else:
            verdict = asyncio.run(
                review_draft(response.answer, retrieved_context)
            )
            final_answer = verdict.final_answer
            review = verdict.model_dump()

        return {
            "answer": final_answer,
            "mode": "MOCK_LLM",
            "fallback": response.fallback or (
                not response.sources and not response.ticket
            ),
            "sources": response.sources,
            "tools_used": response.tools_used,
            "ticket": (
                response.ticket.model_dump()
                if response.ticket is not None
                else None
            ),
            "question": safe_question,
            "governance": decision.to_dict(),
            "review": review,
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The CrewAI support request could not be processed.",
        )


@app.post("/add-document")
def add_document(request: AddDocumentRequest):
    """Add a Markdown policy document and refresh cached RAG components."""

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        request.title.lower(),
    ).strip("_")

    if not slug:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid document title.",
        )

    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = KNOWLEDGE_BASE_DIR / f"{slug}.md"

    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail="A document with this title already exists.",
        )

    try:
        file_path.write_text(
            f"# {request.title.strip()}\n\n"
            f"{request.content.strip()}\n",
            encoding="utf-8",
        )

        # Reset the workflow's cached RAG components so that a later
        # policy search can reload the knowledge base.
        import mock_workflow

        mock_workflow._rag_components = None

        return {
            "message": "Document added successfully.",
            "filename": file_path.name,
            "rag_refresh": "Scheduled for the next policy search",
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The document could not be added.",
        )


@app.websocket("/ws")
async def support_websocket(websocket: WebSocket):
    """Accept support questions over WebSocket until disconnected."""

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            trace_id = str(uuid.uuid4())
            started = time.perf_counter()

            try:
                request = AskRequest.model_validate(data)

                # Run synchronous RAG/ticket processing in a worker thread.
                # This allows the async WebSocket event loop to keep running.
                result = await run_in_threadpool(
                    run_support_workflow,
                    request.question,
                    session_id=request.session_id,
                )

                log_request(trace_id, "/ws", started, "success", 200)
                await websocket.send_json(result)

            except WebSocketDisconnect:
                raise

            except Exception:
                log_request(trace_id, "/ws", started, "error", 400)
                await websocket.send_json(
                    {
                        "error": (
                            "The support request could not be processed."
                        ),
                        "trace_id": trace_id,
                    }
                )

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
