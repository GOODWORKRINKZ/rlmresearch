"""FastAPI server with OpenAI-compatible endpoints."""

import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rlm_assistant.config import get_settings
from rlm_assistant.rlm_client import chat, get_rlm

logger = logging.getLogger(__name__)

# --- Pydantic models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# --- FastAPI app ---

app = FastAPI(title="RLM Dev Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Eagerly initialize RLM singleton on startup."""
    settings = get_settings()
    logger.info(
        "RLM server starting with DeepSeek backend: model=%s",
        settings.deepseek_model,
    )
    get_rlm()
    logger.info("RLM singleton initialized")


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "ok",
        "backend": "deepseek",
        "model": settings.deepseek_model,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "data": [
            {
                "id": "rlm-dev-assistant",
                "object": "model",
                "created": 0,
                "owned_by": "rlm",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Extracts the last user message and sends it to RLM.
    Returns full response (streaming is Phase 4).
    """
    # Extract last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        return {"error": "No user message found"}

    last_message = user_messages[-1].content
    logger.info("Chat request: model=%s, message=%s...", request.model, last_message[:100])

    try:
        response_text = chat(last_message)
    except Exception as e:
        logger.error("RLM chat failed: %s", str(e), exc_info=True)
        return {"error": str(e)}

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
    ).model_dump()


def main():
    """CLI entrypoint for uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
