"""FastAPI server — OpenAI-compatible proxy powered by RLM orchestrator.

ALL requests route through RLM (Recursive Language Model) as the brain:
1. **RLM tool-calling path** (tools in request): RLM processes → may call vscode_* tools →
   returns tool_calls to VS Code → VS Code executes → sends results back → RLM continues
2. **RLM tool results path** (tool results): Feed results back to RLM → continue reasoning
3. **RLM text path** (no tools): RLM processes → returns text response

RLM uses custom tools (local: read_file, write_file, etc.) and deferred VS Code tools
(vscode_run_terminal, vscode_edit_file, etc.) that get translated to OpenAI tool_calls.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rlm_assistant.config import get_settings
from rlm_assistant.rlm_client import (
    chat as rlm_chat,
    clear_vscode_tool_calls,
    get_rlm,
    get_vscode_id_map,
    get_vscode_tool_calls,
)
from rlm_assistant.tools import (
    RLM_TO_VSCODE_TOOL_NAME,
    format_tool_results_for_rlm,
    map_rlm_tool_call_to_vscode,
)

# Configure logging — only set format if no handlers exist (avoid conflict with uvicorn)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger("rlm.server")
logger.setLevel(logging.INFO)

# RLM singleton lock — only one completion() at a time (persistent REPL state is shared)
_rlm_lock = asyncio.Lock()
# Feature flag: set USE_RLM=false to fall back to direct API proxy
USE_RLM = os.getenv("USE_RLM", "true").lower() == "true"

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
    consultant = settings.mimo_model if settings.mimo_api_key else "none"
    logger.info(
        "RLM server starting: root=%s/%s, sub=%s/%s, consultant=%s (custom tool)",
        settings.root_provider,
        settings.active_model_name,
        settings.sub_provider,
        settings.sub_model_name,
        consultant,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    consultant = settings.mimo_model if settings.mimo_api_key else None
    return {
        "status": "ok",
        "root_provider": settings.root_provider,
        "root_model": settings.active_model_name,
        "sub_provider": settings.sub_provider,
        "sub_model": settings.sub_model_name,
        "consultant_model": consultant,
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
    # Map unknown model names (e.g. "rlm-dev-assistant") to active model
    known_models = {"deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2.5-pro"}
    model = request.model if request.model in known_models else _get_model_name()

    has_tools = bool(request.tools)
    has_tool_results = any(m.role == "tool" for m in request.messages)

    # Log request summary
    tool_names = [t["function"]["name"] for t in request.tools] if request.tools else []
    msg_roles = [m.role for m in request.messages]
    logger.info(
        ">>> Request: model=%s stream=%s tools=%s msgs=%d %s",
        request.model, request.stream,
        tool_names if tool_names else "none",
        len(request.messages), msg_roles,
    )

    if has_tool_results and USE_RLM:
        logger.info("RLM tool results path: stream=%s", request.stream)
        return await _handle_tool_results_via_rlm(
            completion_id, created, model, request
        )
    elif has_tools and USE_RLM:
        logger.info("RLM tool-calling path: %d tools, stream=%s", len(request.tools), request.stream)
        return await _handle_tool_calling_via_rlm(
            completion_id, created, model, request
        )
    elif USE_RLM:
        logger.info("RLM text path: stream=%s", request.stream)
        return await _handle_text_via_rlm(
            completion_id, created, model, request
        )
    else:
        # Direct API fallback (USE_RLM=false)
        if has_tool_results:
            logger.info("Direct tool results path")
            return _handle_tool_results_direct(completion_id, created, model, request)
        elif has_tools:
            logger.info("Direct tool-calling path: %d tools", len(request.tools))
            return _handle_tool_calling_direct(completion_id, created, model, request)
        else:
            logger.info("Direct text path")
            return _handle_text_request_direct(completion_id, created, model, request)


def _handle_tool_calling_direct(
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


def _handle_tool_results_direct(
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


def _handle_text_request_direct(
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


# =============================================================================
# RLM-based request handlers — route ALL requests through RLM orchestrator
# =============================================================================


def _strip_vscode_context(text: str) -> str:
    """Remove VS Code internal context blocks from message content.

    VS Code injects <context>, <editorContext>, <reminderInstructions>, <userRequest>
    blocks into messages. These bloat RLM context and confuse the model.
    Extract just the <userRequest> if present, otherwise strip the blocks.
    """
    import re
    # Try to extract user request first
    user_req = re.search(r'<userRequest>(.*?)</userRequest>', text, re.DOTALL)
    if user_req:
        return user_req.group(1).strip()

    # Otherwise strip all VS Code blocks
    text = re.sub(r'<context>.*?</context>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<editorContext>.*?</editorContext>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<reminderInstructions>.*?</reminderInstructions>\s*', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_latest_user_message(messages: list[ChatMessage]) -> str:
    """Extract the latest user message content for RLM."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return _strip_vscode_context(msg.content)
    return ""


def _extract_all_user_messages(messages: list[ChatMessage]) -> str:
    """Extract all user messages as a single text for RLM context."""
    parts = []
    for msg in messages:
        if msg.role == "user" and msg.content:
            cleaned = _strip_vscode_context(msg.content)
            if cleaned:
                parts.append(cleaned)
    return "\n\n".join(parts) if parts else ""


async def _run_rlm_in_thread(message: str) -> str:
    """Run RLM completion in a thread (it's synchronous and blocking).

    Uses the async _rlm_lock to ensure only one RLM runs at a time.
    Returns the response string.
    """
    rlm = get_rlm()
    async with _rlm_lock:
        result = await asyncio.to_thread(rlm.completion, message)
    return result.response


async def _handle_tool_calling_via_rlm(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
):
    """RLM tool-calling path: route through RLM, collect vscode tool calls.

    Flow:
    1. Extract user message from conversation
    2. Clear previous vscode tool calls
    3. Run RLM completion
    4. If RLM called vscode tools → return tool_calls to VS Code
    5. If no vscode tools → return RLM text response
    """
    user_message = _extract_latest_user_message(request.messages)
    if not user_message:
        user_message = _extract_all_user_messages(request.messages)

    logger.info("RLM processing: %.100s...", user_message)

    # Clear previous tool calls
    clear_vscode_tool_calls()

    try:
        # Run RLM (blocking, in thread)
        rlm_response = await _run_rlm_in_thread(user_message)

        # Check for vscode tool calls
        rlm_tool_calls = get_vscode_tool_calls()
        rlm_id_map = get_vscode_id_map()

        if rlm_tool_calls:
            # Convert RLM tool calls to OpenAI format for VS Code
            openai_tool_calls = [map_rlm_tool_call_to_vscode(tc) for tc in rlm_tool_calls]
            logger.info(
                "RLM requested %d vscode tool calls: %s",
                len(openai_tool_calls),
                [tc["function"]["name"] for tc in openai_tool_calls],
            )

            # Store the id_map for when VS Code sends results back
            _pending_id_map.update(rlm_id_map)

            if request.stream:
                return _stream_rlm_tool_calls(
                    completion_id, created, model, openai_tool_calls, content=rlm_response
                )
            else:
                return _format_rlm_tool_calls_response(
                    completion_id, created, model, openai_tool_calls, rlm_response
                )
        else:
            # No vscode tool calls — return RLM's text response
            logger.info("RLM text response: %.100s...", rlm_response)
            if request.stream:
                return _stream_rlm_text(completion_id, created, model, rlm_response)
            else:
                return _format_rlm_text_response(completion_id, created, model, rlm_response)

    except Exception as e:
        logger.error("RLM tool-calling failed: %s", str(e), exc_info=True)
        return _error_response(completion_id, created, model, str(e))


async def _handle_tool_results_via_rlm(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
):
    """RLM tool results path: feed VS Code tool results back to RLM.

    Flow:
    1. Extract tool results from VS Code messages
    2. Format as a message for RLM
    3. Run RLM completion with results
    4. If RLM calls more vscode tools → return tool_calls
    5. If no more tools → return RLM text response
    """
    # Build tool results text for RLM
    tool_results_text = format_tool_results_for_rlm(request.messages, _pending_id_map)

    logger.info("Feeding %d tool results to RLM", sum(1 for m in request.messages if m.role == "tool"))

    # Clear previous tool calls before new RLM run
    clear_vscode_tool_calls()

    try:
        # Run RLM with tool results
        rlm_response = await _run_rlm_in_thread(tool_results_text)

        # Check for more vscode tool calls
        rlm_tool_calls = get_vscode_tool_calls()
        rlm_id_map = get_vscode_id_map()

        if rlm_tool_calls:
            openai_tool_calls = [map_rlm_tool_call_to_vscode(tc) for tc in rlm_tool_calls]
            logger.info(
                "RLM requested %d more vscode tool calls: %s",
                len(openai_tool_calls),
                [tc["function"]["name"] for tc in openai_tool_calls],
            )
            _pending_id_map.update(rlm_id_map)

            if request.stream:
                return _stream_rlm_tool_calls(
                    completion_id, created, model, openai_tool_calls, content=rlm_response
                )
            else:
                return _format_rlm_tool_calls_response(
                    completion_id, created, model, openai_tool_calls, rlm_response
                )
        else:
            logger.info("RLM final response after tool results: %.100s...", rlm_response)
            if request.stream:
                return _stream_rlm_text(completion_id, created, model, rlm_response)
            else:
                return _format_rlm_text_response(completion_id, created, model, rlm_response)

    except Exception as e:
        logger.error("RLM tool results processing failed: %s", str(e), exc_info=True)
        return _error_response(completion_id, created, model, str(e))


async def _handle_text_via_rlm(
    completion_id: str,
    created: int,
    model: str,
    request: ChatCompletionRequest,
):
    """RLM text path: process text-only request through RLM.

    Used when VS Code sends a message without tools (e.g., follow-up text).
    """
    user_message = _extract_latest_user_message(request.messages)
    if not user_message:
        user_message = _extract_all_user_messages(request.messages)

    logger.info("RLM text processing: %.100s...", user_message)

    clear_vscode_tool_calls()

    try:
        rlm_response = await _run_rlm_in_thread(user_message)

        # Check if RLM requested vscode tools (unlikely for text path, but possible)
        rlm_tool_calls = get_vscode_tool_calls()

        if rlm_tool_calls:
            openai_tool_calls = [map_rlm_tool_call_to_vscode(tc) for tc in rlm_tool_calls]
            _pending_id_map.update(get_vscode_id_map())
            logger.info("RLM text path unexpectedly got %d tool calls", len(openai_tool_calls))
            if request.stream:
                return _stream_rlm_tool_calls(
                    completion_id, created, model, openai_tool_calls, content=rlm_response
                )
            else:
                return _format_rlm_tool_calls_response(
                    completion_id, created, model, openai_tool_calls, rlm_response
                )

        logger.info("RLM text response: %.100s...", rlm_response)
        if request.stream:
            return _stream_rlm_text(completion_id, created, model, rlm_response)
        else:
            return _format_rlm_text_response(completion_id, created, model, rlm_response)

    except Exception as e:
        logger.error("RLM text processing failed: %s", str(e), exc_info=True)
        return _error_response(completion_id, created, model, str(e))


# --- Pending tool call ID map (stores mapping between RLM and VS Code tool call IDs) ---
# This is populated when RLM returns tool calls and consumed when VS Code sends results back.
_pending_id_map: dict[str, dict] = {}


# --- RLM response formatters ---


def _stream_rlm_text(
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> StreamingResponse:
    """Stream RLM text response as SSE chunks."""
    def generate():
        try:
            # Role delta
            yield _sse_chunk(completion_id, created, model, {"role": "assistant"}, None)

            # Stream text in word-sized chunks for smooth UX
            if text:
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield _sse_chunk(completion_id, created, model, {"content": chunk}, None)

            yield _sse_chunk(completion_id, created, model, {}, "stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream RLM text failed: %s", str(e), exc_info=True)
            yield _sse_chunk(completion_id, created, model, {"content": f"Error: {e}"}, "stop")
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _stream_rlm_tool_calls(
    completion_id: str,
    created: int,
    model: str,
    tool_calls: list[dict],
    content: str = "",
) -> StreamingResponse:
    """Stream RLM tool_calls as SSE deltas (OpenAI format).

    Emits:
    1. Role delta
    2. Optional content delta (RLM's reasoning text)
    3. Tool call deltas: first has id+name, then argument chunks
    4. finish_reason: "tool_calls"
    """
    def generate():
        try:
            # Role delta
            yield _sse_chunk(completion_id, created, model, {"role": "assistant"}, None)

            # Content delta (RLM's reasoning text, if any)
            if content:
                yield _sse_chunk(completion_id, created, model, {"content": content}, None)

            # Tool call deltas
            for i, tc in enumerate(tool_calls):
                fn = tc["function"]
                # First delta for this tool: id, type, name
                yield _sse_chunk(completion_id, created, model, {
                    "tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": fn["name"]},
                    }]
                }, None)

                # Arguments delta (stream in chunks for proper SSE format)
                args_str = fn["arguments"]
                chunk_size = 50
                for j in range(0, len(args_str), chunk_size):
                    yield _sse_chunk(completion_id, created, model, {
                        "tool_calls": [{
                            "index": i,
                            "function": {"arguments": args_str[j:j + chunk_size]},
                        }]
                    }, None)

            # Finish
            yield _sse_chunk(completion_id, created, model, {}, "tool_calls")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream RLM tool_calls failed: %s", str(e), exc_info=True)
            yield _sse_chunk(completion_id, created, model, {"content": f"Error: {e}"}, "stop")
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _format_rlm_text_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> dict:
    """Format RLM text response as non-streaming OpenAI completion."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def _format_rlm_tool_calls_response(
    completion_id: str,
    created: int,
    model: str,
    tool_calls: list[dict],
    content: str = "",
) -> dict:
    """Format RLM tool_calls as non-streaming OpenAI completion."""
    msg: dict[str, Any] = {"role": "assistant"}
    if content:
        msg["content"] = content
    msg["tool_calls"] = tool_calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls",
        }],
    }


def _error_response(
    completion_id: str,
    created: int,
    model: str,
    error: str,
) -> dict:
    """Return an error response in OpenAI format."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Error: {error}",
            },
            "finish_reason": "stop",
        }],
    }
