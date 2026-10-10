"""FastAPI interface for the Ola Support Agent with safe JSONL logging."""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

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

        for key in ("trace_id", "endpoint", "duration_ms", "outcome", "status_code"):
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
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "outcome": outcome,
            "status_code": status_code,
        },
    )


app = FastAPI(
    title="Ola Support Agent API",
    description="Local mock support workflow with RAG and ticket lookup.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


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
    """Answer a support question and log safe request metadata."""

    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        result = run_support_workflow(
            request.question,
            session_id=request.session_id,
        )
        log_request(trace_id, "/ask", started, "success", 200)
        return result

    except Exception:
        log_request(trace_id, "/ask", started, "error", 500)
        raise HTTPException(
            status_code=500,
            detail="The support request could not be processed.",
        )


@app.post("/add-document")
def add_document(request: AddDocumentRequest):
    """Add a Markdown policy document and refresh cached RAG components."""

    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        request.title.lower(),
    ).strip("_")

    if not slug:
        log_request(trace_id, "/add-document", started, "invalid_title", 400)
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid document title.",
        )

    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = KNOWLEDGE_BASE_DIR / f"{slug}.md"

    if file_path.exists():
        log_request(trace_id, "/add-document", started, "duplicate", 409)
        raise HTTPException(
            status_code=409,
            detail="A document with this title already exists.",
        )

    try:
        file_path.write_text(
            f"# {request.title.strip()}\n\n{request.content.strip()}\n",
            encoding="utf-8",
        )

        import crew_support
        crew_support._rag_components = None

        log_request(trace_id, "/add-document", started, "success", 200)

        return {
            "message": "Document added successfully.",
            "filename": file_path.name,
            "rag_refresh": "Scheduled for the next policy search",
        }

    except Exception:
        log_request(trace_id, "/add-document", started, "error", 500)
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
                result = run_support_workflow(
                    request.question,
                    session_id=request.session_id,
                )

                log_request(trace_id, "/ws", started, "success", 200)
                await websocket.send_json(result)

            except Exception:
                log_request(trace_id, "/ws", started, "error", 400)
                await websocket.send_json({
                    "error": "The support request could not be processed.",
                    "trace_id": trace_id,
                })

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
