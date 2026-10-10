
"""FastAPI interface for the Ola Support Agent."""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from mock_workflow import run_support_workflow


app = FastAPI(
    title="Ola Support Agent API",
    description="Local mock support workflow with RAG and ticket lookup.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


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
    """Answer a support question using the local workflow."""

    try:
        return run_support_workflow(
            request.question,
            session_id=request.session_id,
        )
    except Exception:
        # Avoid exposing internal exception details to API clients.
        raise HTTPException(
            status_code=500,
            detail="The support request could not be processed.",
        )


@app.post("/add-document")
def add_document(request: AddDocumentRequest):
    """Add a Markdown policy document and refresh cached RAG components."""

    # Convert the title into a safe filename.
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

    file_path.write_text(
        f"# {request.title.strip()}\n\n{request.content.strip()}\n",
        encoding="utf-8",
    )

    # Refresh the cached RAG components so the new document can be indexed.
    # Rebuilding occurs on the next policy search.
    import crew_support
    crew_support._rag_components = None

    return {
        "message": "Document added successfully.",
        "filename": file_path.name,
        "rag_refresh": "Scheduled for the next policy search",
    }


@app.websocket("/ws")
async def support_websocket(websocket: WebSocket):
    """Accept support questions over WebSocket until disconnected."""

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            try:
                request = AskRequest.model_validate(data)
                result = run_support_workflow(
                    request.question,
                    session_id=request.session_id,
                )
                await websocket.send_json(result)

            except Exception:
                await websocket.send_json({
                    "error": "The support request could not be processed."
                })

    except WebSocketDisconnect:
        # A disconnected client is a normal event, not a server failure.
        print("WebSocket client disconnected gracefully.")
