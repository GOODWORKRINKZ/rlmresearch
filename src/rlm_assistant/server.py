"""FastAPI server with OpenAI-compatible endpoints."""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    # VS Code Copilot sends these — parsed but NOT forwarded to RLM
    tools: Optional[list] = None
    tool_choice: Optional[str] = None


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




def sanitize_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Strip tool-role messages that RLM doesn't understand.

    VS Code Copilot sends OpenAI-style tool definitions (tools: [...], tool_choice: "auto")
    when toolCalling: true in chatLanguageModels.json. RLM doesn't support OpenAI function
    calling — it uses REPL blocks instead. We parse these fields to avoid 422 errors, but
    strip them before passing to RLM to prevent model confusion.
    """
    return [m for m in messages if m.role in ("system", "user", "assistant")]


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
        "RLM server starting: pro_model=%s, flash_model=%s",
        settings.deepseek_pro_model,
        settings.deepseek_flash_model,
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
        "pro_model": settings.deepseek_pro_model,
        "flash_model": settings.deepseek_flash_model,
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
    Supports both streaming (SSE) and non-streaming responses.
    """
    # Strip tool definitions from VS Code requests
    if request.tools:
        logger.info("Stripping %d tool definitions from VS Code request", len(request.tools))

    # Sanitize messages — remove tool-role messages RLM doesn't understand
    clean_messages = sanitize_messages(request.messages)

    # Extract last user message
    user_messages = [m for m in clean_messages if m.role == "user"]
    if not user_messages:
        return {"error": "No user message found"}

    last_message = user_messages[-1].content
    logger.info("Chat request: model=%s, stream=%s, message=%s...", request.model, request.stream, last_message[:100])

    try:
        response_text = chat(last_message)
    except Exception as e:
        logger.error("RLM chat failed: %s", str(e), exc_info=True)
        return {"error": str(e)}

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        # SSE streaming: send full response as one chunk, then [DONE]
        def generate():
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            chunk["choices"][0]["delta"] = {"content": response_text}
            yield f"data: {json.dumps(chunk)}\n\n"

            chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["finish_reason"] = "stop"
            yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
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
