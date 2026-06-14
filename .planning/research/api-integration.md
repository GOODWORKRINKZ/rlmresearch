# API Integration Research: DeepSeek + Mimo for RLM

**Researched:** 2026-06-14
**Confidence:** HIGH (DeepSeek — official docs verified), MEDIUM (Mimo — HuggingFace + SiliconFlow docs verified, exact model name needs runtime confirmation)

---

## 1. DeepSeek API

### Overview

DeepSeek provides an OpenAI-compatible API at `https://api.deepseek.com`. Two model families are available:

| Model | Model Name | Thinking Mode | Context | Max Output | Concurrency |
|-------|-----------|---------------|---------|------------|-------------|
| V4 Flash | `deepseek-v4-flash` | Both (default: thinking) | 1M | 384K | 2500 |
| V4 Pro | `deepseek-v4-pro` | Both (default: thinking) | 1M | 384K | 500 |

**⚠️ Deprecation Notice:** `deepseek-chat` and `deepseek-reasoner` will be deprecated on **2026/07/24 15:59 UTC**. They map to non-thinking and thinking modes of `deepseek-v4-flash` respectively. Use `deepseek-v4-flash` and `deepseek-v4-pro` instead.

### Authentication

```python
# Header-based auth
Authorization: Bearer ${DEEPSEEK_API_KEY}
```

### Base URL

```
OpenAI format:    https://api.deepseek.com
Anthropic format: https://api.deepseek.com/anthropic
```

**Note:** No `/v1` suffix needed — the OpenAI SDK adds it automatically when you set `base_url="https://api.deepseek.com"`.

### Pricing (per 1M tokens)

| | V4 Flash | V4 Pro |
|---|----------|--------|
| Input (cache hit) | **$0.0028** | **$0.003625** |
| Input (cache miss) | $0.14 | $0.435 |
| Output | $0.28 | $0.87 |

**Cache hit discount:** ~50x cheaper on input tokens. This is automatic — no code changes needed.

### Context Caching (Automatic)

DeepSeek's disk-based context caching is **enabled by default for all users**. No code changes required.

**How it works:**
1. Each request creates cache prefix units at user input end and model output end
2. Subsequent requests matching those prefixes get cache hits
3. System auto-detects common prefixes across requests and persists them
4. Long inputs get cache units at fixed token intervals

**Cache hit rules:**
- Prefix must **fully match** a persisted cache unit (Sliding Window Attention)
- Cache builds in seconds, clears in hours to days (best-effort)
- Check `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens` in response

**Example — multi-round conversation:**
```
Round 1: [system, user] → cache miss
Round 2: [system, user, assistant, user2] → cache hit on [system, user]
Round 3: [system, user, assistant, user2, assistant2, user3] → cache hit on [system, user, assistant, user2]
```

### Thinking Mode

Both V4 Flash and V4 Pro support thinking mode (Chain of Thought). The API returns:
- `reasoning_content`: CoT tokens
- `content`: final answer

**For V4 Flash:** thinking is the default mode. To disable, use non-thinking mode.

**Important:** When doing multi-round conversations, strip `reasoning_content` from previous assistant messages before sending — the API returns 400 if `reasoning_content` is in input messages.

### Features

| Feature | V4 Flash | V4 Pro |
|---------|----------|--------|
| JSON Output | ✓ | ✓ |
| Tool Calls | ✓ | ✓ |
| Chat Prefix Completion (Beta) | ✓ | ✓ |
| FIM Completion (Beta) | Non-thinking only | Non-thinking only |

**Not supported in thinking mode:** `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs` (silently ignored except logprobs which errors).

---

## 2. Mimo API (via SiliconFlow)

### Overview

Mimo is Xiaomi's reasoning model family. Available via SiliconFlow (Chinese cloud AI provider with OpenAI-compatible API).

| Model | HuggingFace ID | Parameters | Status |
|-------|---------------|------------|--------|
| MiMo-V2.5-Pro | `XiaomiMiMo/MiMo-V2.5-Pro` | 1T | Latest (May 2026) |
| MiMo-V2.5 | `XiaomiMiMo/MiMo-V2.5` | 311B | Latest (May 2026) |
| MiMo-V2.5-Pro-FP4-DFlash | `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` | 554B | 6 days ago |
| MiMo-V2-Flash | `XiaomiMiMo/MiMo-V2-Flash` | 310B | Apr 2026 |

### SiliconFlow API

**Base URL:**
```
https://api.siliconflow.cn/v1
```

**Authentication:**
```
Authorization: Bearer YOUR_API_KEY
```

**Endpoint:**
```
POST https://api.siliconflow.cn/v1/chat/completions
```

**Model name format on SiliconFlow:**
The SiliconFlow API uses a format like `Pro/zai-org/GLM-4.7` for their models. Based on the model naming pattern, Mimo models are likely:
- `XiaomiMiMo/MiMo-V2.5-Pro` or similar
- **⚠️ NEEDS RUNTIME CONFIRMATION** — exact model name must be verified by calling `GET https://api.siliconflow.cn/v1/models` or checking the SiliconFlow dashboard.

**SiliconFlow features:**
- OpenAI-compatible chat completions format
- Streaming support (SSE)
- Tool calling support
- `enable_thinking` parameter for reasoning models
- `thinking_budget` parameter (128–32768 tokens)
- `reasoning_effort` parameter (for DeepSeek models on SiliconFlow)
- Supports `response_format` for JSON output

**Example request:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_SILICONFLOW_KEY",
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2.5-Pro",  # verify exact name
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ],
    stream=False
)
```

**Response format:**
```json
{
  "id": "...",
  "object": "chat.completion",
  "created": 1768899826,
  "model": "XiaomiMiMo/MiMo-V2.5-Pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...",
        "reasoning_content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 1540,
    "total_tokens": 1555,
    "completion_tokens_details": {
      "reasoning_tokens": 1190
    },
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```

---

## 3. OpenAI-Compatible API Patterns

### FastAPI Server for OpenAI Compatibility

To create a proxy server that wraps DeepSeek/Mimo behind an OpenAI-compatible endpoint (useful for RLM):

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import json
import time
import uuid

app = FastAPI()

# Model routing
MODELS = {
    "deepseek-v4-flash": {
        "client": OpenAI(api_key="...", base_url="https://api.deepseek.com"),
        "model": "deepseek-v4-flash"
    },
    "deepseek-v4-pro": {
        "client": OpenAI(api_key="...", base_url="https://api.deepseek.com"),
        "model": "deepseek-v4-pro"
    },
    "mimo-v2.5-pro": {
        "client": OpenAI(api_key="...", base_url="https://api.siliconflow.cn/v1"),
        "model": "XiaomiMiMo/MiMo-V2.5-Pro"
    }
}

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None
    # ... other OpenAI params

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    provider = MODELS.get(request.model)
    if not provider:
        return {"error": f"Model {request.model} not found"}, 404
    
    if request.stream:
        return StreamingResponse(
            stream_response(provider, request),
            media_type="text/event-stream"
        )
    
    response = provider["client"].chat.completions.create(
        model=provider["model"],
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=False
    )
    return response.to_dict()

async def stream_response(provider, request):
    stream = provider["client"].chat.completions.create(
        model=provider["model"],
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=True
    )
    for chunk in stream:
        data = json.dumps(chunk.to_dict())
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"
```

### Key OpenAI-Compatible Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Main chat endpoint |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Embeddings (if supported) |

### Streaming (SSE) Format

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0}]}

data: [DONE]
```

### VS Code Copilot Custom Endpoint Configuration

VS Code supports custom OpenAI-compatible endpoints via `chatLanguageModels.json`:

```json
[
  {
    "name": "RLM Server",
    "vendor": "customendpoint",
    "apiKey": "your-rlm-server-key",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash (via RLM)",
        "url": "http://localhost:8000/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 1000000,
        "maxOutputTokens": 384000,
        "thinking": true,
        "streaming": true
      },
      {
        "id": "mimo-v2.5-pro",
        "name": "MiMo V2.5 Pro (via RLM)",
        "url": "http://localhost:8000/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 131072,
        "maxOutputTokens": 32768,
        "thinking": true,
        "streaming": true
      }
    ]
  }
]
```

**Configuration location:** `~/.config/Code/User/chatLanguageModels.json` or via Command Palette → `Chat: Manage Language Models` → Add Models → Custom Endpoint.

**Key properties:**
- `vendor`: `"customendpoint"` for any OpenAI-compatible API
- `apiType`: `"chat-completions"` (also supports `"responses"` and `"messages"`)
- `toolCalling`: Must be `true` for agent mode in VS Code
- `thinking`: Set `true` if model supports reasoning/thinking
- `supportsReasoningEffort`: Array of effort levels, e.g. `["low", "medium", "high"]`
- `reasoningEffortFormat`: `"chat-completions"` sends top-level `reasoning_effort` string

**Direct DeepSeek integration (no proxy needed):**
```json
[
  {
    "name": "DeepSeek",
    "vendor": "customendpoint",
    "apiKey": "sk-...",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "toolCalling": true,
        "maxInputTokens": 1000000,
        "maxOutputTokens": 384000,
        "thinking": true,
        "streaming": true,
        "supportsReasoningEffort": ["high", "max"]
      }
    ]
  }
]
```

---

## 4. Cost Optimization

### Cost Comparison Table

| Scenario | V4 Flash | V4 Pro | Ratio |
|----------|----------|--------|-------|
| 1M input tokens (cache miss) | $0.14 | $0.435 | Pro 3.1x more |
| 1M input tokens (cache hit) | $0.0028 | $0.003625 | Pro 1.3x more |
| 1M output tokens | $0.28 | $0.87 | Pro 3.1x more |
| 100K input + 10K output (cache miss) | $0.017 | $0.052 | Pro 3.1x more |
| 100K input + 10K output (cache hit) | $0.0031 | $0.012 | Pro 3.9x more |

### When to Use Each Model

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Simple code completion | V4 Flash | Fast, cheap, good enough |
| Complex reasoning/planning | V4 Pro | Better reasoning quality |
| Recursive sub-calls (depth > 0) | V4 Flash | Cost control for repeated calls |
| Initial task decomposition | V4 Pro | Accuracy matters for planning |
| Bulk text processing | V4 Flash | Cost-effective for high volume |
| Agent tool-calling loops | V4 Flash | Many iterations = many calls |

### RLM Recursive Call Cost Analysis

RLM's recursive structure means costs multiply with depth:

```
Depth 0 (main): 1 call to primary model
Depth 1 (sub):  N calls to sub-model (N = number of subtasks)
Depth 2 (sub-sub): M calls per sub-subtask
```

**Example cost for a 3-level recursive task with 3 subtasks per level:**

Using V4 Pro at all levels (cache miss):
```
Level 0: 1 × ($0.435/1M × 10K input + $0.87/1M × 5K output) = $0.0087
Level 1: 3 × ($0.435/1M × 8K input + $0.87/1M × 3K output) = $0.018
Level 2: 9 × ($0.435/1M × 5K input + $0.87/1M × 2K output) = $0.035
Total: ~$0.062
```

Using V4 Pro at level 0, V4 Flash at levels 1-2:
```
Level 0: 1 × ($0.435/1M × 10K input + $0.87/1M × 5K output) = $0.0087
Level 1: 3 × ($0.14/1M × 8K input + $0.28/1M × 3K output) = $0.006
Level 2: 9 × ($0.14/1M × 5K input + $0.28/1M × 2K output) = $0.011
Total: ~$0.026
```

**Savings: ~58% by using Flash for sub-calls.**

### Cache Optimization Strategies for RLM

1. **Consistent system prompts:** Keep the system message identical across calls to maximize prefix cache hits
2. **Structured conversation history:** Append new turns rather than rebuilding messages
3. **Shared context for sub-calls:** If sub-calls share a common prefix (task description), DeepSeek will auto-cache it after 2 requests
4. **Batch related sub-calls:** Make sub-calls with similar prefixes close together in time

### Pricing on SiliconFlow (for Mimo)

SiliconFlow pricing is separate from DeepSeek. Check `https://cloud.siliconflow.cn/pricing` for current rates. SiliconFlow also supports caching for some models.

---

## 5. Integration with RLM Library

### RLM Configuration for DeepSeek + Mimo

Based on the RLM library research (see `rlm-integration.md`):

```python
import os
from rlm import RLM

# Option A: DeepSeek primary, Mimo for sub-calls
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "model_name": "XiaomiMiMo/MiMo-V2.5-Pro",  # verify exact name
        "base_url": "https://api.siliconflow.cn/v1",
    }],
)

# Option B: Mimo primary, DeepSeek for sub-calls
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "model_name": "XiaomiMiMo/MiMo-V2.5-Pro",
        "base_url": "https://api.siliconflow.cn/v1",
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-v4-flash",  # cheaper for sub-calls
        "base_url": "https://api.deepseek.com",
    }],
)

# Option C: All DeepSeek (Pro primary, Flash sub-calls) — simplest
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    }],
)
```

### RLM as OpenAI-Compatible Server

To expose RLM as an OpenAI-compatible endpoint for VS Code Copilot:

```python
from fastapi import FastAPI
from rlm import RLM
import os

app = FastAPI()
rlm = RLM(...)  # configure as above

@app.post("/v1/chat/completions")
async def chat_completions(request):
    # Route through RLM's recursive logic
    result = rlm.query(request.messages)
    return format_as_openai_response(result)
```

---

## 6. Action Items & Open Questions

### Must-Verify (Runtime)

| Item | How to Verify | Priority |
|------|---------------|----------|
| SiliconFlow Mimo model exact name | Call `GET /v1/models` on SiliconFlow | HIGH |
| SiliconFlow Mimo pricing | Check SiliconFlow dashboard | HIGH |
| SiliconFlow streaming format | Test with curl | MEDIUM |
| DeepSeek V4 thinking mode parameter | Test `thinking` param in request | MEDIUM |
| RLM `other_backends` len validation | Check if current code still validates len==1 | MEDIUM |

### Verified

| Item | Status | Source |
|------|--------|--------|
| DeepSeek base URL | ✓ `https://api.deepseek.com` | Official docs |
| DeepSeek pricing | ✓ Verified | Official pricing page |
| DeepSeek context caching | ✓ Automatic, no code changes | Official docs |
| DeepSeek model names | ✓ `deepseek-v4-flash`, `deepseek-v4-pro` | Official docs |
| DeepSeek deprecation | ✓ `deepseek-chat`/`deepseek-reasoner` deprecated 2026/07/24 | Official docs |
| SiliconFlow base URL | ✓ `https://api.siliconflow.cn/v1` | SiliconFlow docs |
| SiliconFlow OpenAI compat | ✓ Verified (chat completions format) | SiliconFlow API docs |
| VS Code custom endpoint | ✓ `vendor: "customendpoint"` in chatLanguageModels.json | VS Code docs |
| RLM OpenAI backend | ✓ Works with any `base_url` | RLM source code |

---

## Sources

| Source | URL | Confidence |
|--------|-----|------------|
| DeepSeek API Docs | https://api-docs.deepseek.com/ | HIGH |
| DeepSeek Pricing | https://api-docs.deepseek.com/quick_start/pricing | HIGH |
| DeepSeek Context Caching | https://api-docs.deepseek.com/guides/kv_cache | HIGH |
| DeepSeek Reasoning Model | https://api-docs.deepseek.com/guides/reasoning_model | HIGH |
| SiliconFlow API Docs | https://docs.siliconflow.cn/en/api-reference/chat-completions/chat-completions | MEDIUM |
| SiliconFlow OpenAI Compat | https://docs.siliconflow.cn/en/userguide/capabilities/openai-compatibility | MEDIUM |
| Xiaomi MiMo (HuggingFace) | https://huggingface.co/XiaomiMiMo | HIGH |
| VS Code Language Models | https://code.visualstudio.com/docs/copilot/language-models | HIGH |
| OpenAI Python SDK | https://github.com/openai/openai-python | HIGH |
