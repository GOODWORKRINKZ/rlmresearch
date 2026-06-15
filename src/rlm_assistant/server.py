"""FastAPI server with OpenAI-compatible endpoints.

Two response paths:
1. **Tool-calling path** (tools in request): Direct OpenAI API call → returns tool_calls
2. **RLM path** (no tools): RLM orchestrator → returns text response

When VS Code sends tool results back, we build context and use RLM to synthesize
the final answer.
"""

import json
import logging
import time
import uuid
from typing import Any, Optional

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rlm_assistant.config import get_settings

logger = logging.getLogger(__name__)

# --- Pydantic models ---


class ToolCall(BaseModel):
    """OpenAI-format tool call."""
    id: str
    type: str = "function"
    function: dict[str, Any]


class ChatMessage(BaseModel):
    """OpenAI-format chat message."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[str] = None


# --- Direct API client (for tool-calling path) ---

_api_client: Optional[openai.OpenAI] = None


def _get_api_client() -> openai.OpenAI:
    """Get or create the direct OpenAI API client."""
    global _api_client
    if _api_client is None:
        settings = get_settings()
        if settings.active_provider == "mimo":
            _api_client = openai.OpenAI(
                api_key=settings.mimo_api_key,
                base_url=settings.mimo_base_url,
            )
        else:
            _api_client = openai.OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
    return _api_client


def _get_extra_body() -> dict:
    """Get extra_body for DeepSeek API (disable thinking mode)."""
    settings = get_settings()
    if settings.active_provider == "deepseek":
        return {"thinking": {"type": "disabled"}}
    return {}


def _get_model_name() -> str:
    """Get the model name for the active provider."""
    settings = get_settings()
    if settings.active_provider == "mimo":
        return settings.mimo_model
    return settings.deepseek_pro_model


# --- FastAPI app ---

app = FastAPI(title="RLM Dev Assistant", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Log startup info."""
    settings = get_settings()
    logger.info(
        "RLM server starting: provider=%s, model=%s",
        settings.active_provider,
        settings.active_model_name,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.active_provider,
        "model": settings.active_model_name,
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

    Two paths:
    1. Tools present → direct API call → return tool_calls
    2. No tools → RLM orchestrator → return text
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model or _get_model_name()

    has_tools = bool(request.tools)
    has_tool_results = any(m.role == "tool" for m in request.messages)

    if has_tools:
        logger.info("Tool-calling path: %d tools, stream=%s", len(request.tools), request.stream)
        return _handle_tool_calling(
            completion_id, created, model, request
        )
    elif has_tool_results:
        logger.info("Tool results path: stream=%s", request.stream)
        return _handle_tool_results(
            completion_id, created, model, request
        )
    else:
        logger.info("Text path: stream=%s", request.stream)
        return _handle_text_request(
            completion_id, created, model, request
        )


def _handle_tool_calling(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
) -> dict:
    """Handle requests with tools: direct API call to get tool_calls."""
    client = _get_api_client()
    messages = _messages_to_openai(request.messages)

    if request.stream:
        return _stream_tool_calls_response(
            completion_id, created, model, client, messages, request
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=request.tools,
            tool_choice=request.tool_choice or "auto",
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body=_get_extra_body(),
        )

        choice = response.choices[0]
        message = choice.message

        result_msg = {"role": "assistant"}
        if message.content:
            result_msg["content"] = message.content
        if message.tool_calls:
            result_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        finish_reason = choice.finish_reason or "stop"
        if message.tool_calls:
            finish_reason = "tool_calls"

        logger.info(
            "Tool-calling response: finish=%s, tool_calls=%s",
            finish_reason,
            [tc.function.name for tc in (message.tool_calls or [])],
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": result_msg,
                "finish_reason": finish_reason,
            }],
        }

    except Exception as e:
        logger.error("Direct API call failed: %s", str(e), exc_info=True)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error calling API: {str(e)}",
                },
                "finish_reason": "stop",
            }],
        }


def _handle_tool_results(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
) -> dict:
    """Handle tool results: forward conversation with results to direct API."""
    client = _get_api_client()
    messages = _messages_to_openai(request.messages)

    if request.stream:
        return _stream_tool_calls_response(
            completion_id, created, model, client, messages, request
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body=_get_extra_body(),
        )

        choice = response.choices[0]
        message = choice.message

        result_msg = {"role": "assistant"}
        if message.content:
            result_msg["content"] = message.content
        if message.tool_calls:
            result_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        finish_reason = choice.finish_reason or "stop"
        if message.tool_calls:
            finish_reason = "tool_calls"

        logger.info("Tool results response: finish=%s", finish_reason)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": result_msg,
                "finish_reason": finish_reason,
            }],
        }

    except Exception as e:
        logger.error("Tool results API call failed: %s", str(e), exc_info=True)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error processing tool results: {str(e)}",
                },
                "finish_reason": "stop",
            }],
        }


def _stream_tool_calls_response(
    completion_id: str,
    created: int,
    model: str,
    client: openai.OpenAI,
    messages: list[dict],
    request: ChatCompletionRequest,
) -> StreamingResponse:
    """Stream tool-calling response as SSE — handles both tool_calls and text deltas."""
    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=request.tools,
                tool_choice=request.tool_choice or "auto",
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                extra_body=_get_extra_body(),
            )

            sent_role = False
            saw_tool_calls = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                finish = choice.finish_reason

                # Send role delta first
                if not sent_role:
                    yield _sse_chunk(completion_id, created, model, {"role": "assistant"}, None)
                    sent_role = True

                # Content delta
                if delta.content:
                    yield _sse_chunk(completion_id, created, model, {"content": delta.content}, None)

                # Tool calls delta
                if delta.tool_calls:
                    saw_tool_calls = True
                    tc_deltas = []
                    for tc in delta.tool_calls:
                        tc_delta: dict[str, Any] = {"index": tc.index}
                        if tc.id:
                            tc_delta["id"] = tc.id
                            tc_delta["type"] = "function"
                        if tc.function:
                            fn: dict[str, Any] = {}
                            if tc.function.name:
                                fn["name"] = tc.function.name
                            if tc.function.arguments:
                                fn["arguments"] = tc.function.arguments
                            if fn:
                                tc_delta["function"] = fn
                        tc_deltas.append(tc_delta)
                    yield _sse_chunk(completion_id, created, model, {"tool_calls": tc_deltas}, None)

                # Finish — force "tool_calls" if we saw tool_calls
                if finish:
                    final_reason = "tool_calls" if saw_tool_calls else finish
                    yield _sse_chunk(completion_id, created, model, {}, final_reason)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream tool-calls failed: %s", str(e), exc_info=True)
            yield _sse_chunk(completion_id, created, model, {"content": f"Error: {e}"}, "stop")
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _sse_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str],
) -> str:
    """Format a single SSE chunk."""
    return f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"


def _handle_text_request(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
) -> dict:
    """Handle text-only requests: direct API call."""
    client = _get_api_client()
    messages = _messages_to_openai(request.messages)

    if request.stream:
        return _stream_direct_response(
            completion_id, created, model, client, messages, request
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body=_get_extra_body(),
        )

        content = response.choices[0].message.content or ""

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        }

    except Exception as e:
        logger.error("Text API call failed: %s", str(e), exc_info=True)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                },
                "finish_reason": "stop",
            }],
        }


def _messages_to_openai(messages: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage list to OpenAI API format."""
    result = []
    for msg in messages:
        entry = {"role": msg.role}

        if msg.content is not None:
            entry["content"] = msg.content

        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id

        result.append(entry)

    return result


def _stream_direct_response(
    completion_id: str,
    created: int,
    model: str,
    client: openai.OpenAI,
    messages: list[dict],
    request: ChatCompletionRequest,
) -> StreamingResponse:
    """Stream direct API response as SSE (text-only path)."""
    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                extra_body=_get_extra_body(),
            )

            yield _sse_chunk(completion_id, created, model, {"role": "assistant"}, None)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield _sse_chunk(completion_id, created, model, {"content": chunk.choices[0].delta.content}, None)

            yield _sse_chunk(completion_id, created, model, {}, "stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream text failed: %s", str(e), exc_info=True)
            yield _sse_chunk(completion_id, created, model, {"content": f"Error: {e}"}, "stop")
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
