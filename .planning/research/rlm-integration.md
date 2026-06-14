# RLM Library Integration Research

**Source:** [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
**Researched:** 2026-06-14
**Confidence:** HIGH (direct source code analysis + official docs)

---

## 1. Backend & Backend_kwargs — How Custom Providers Work

### Architecture

RLM routes backends via `get_client()` in `rlm/clients/__init__.py`. The routing is a simple if/elif chain:

```python
def get_client(backend: str, backend_kwargs: dict) -> BaseLM:
    if backend == "openai":
        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        return OpenAIClient(**backend_kwargs)  # same class, different base_url
    elif backend == "openrouter":
        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "anthropic":
        return AnthropicClient(**backend_kwargs)
    # ...
```

**Supported backends:** `openai`, `vllm`, `portkey`, `openrouter`, `anthropic`, `azure_openai`, `gemini`, `vercel`

### Key Insight: `base_url` Override

The `OpenAIClient` accepts `base_url` as a constructor parameter. This is the critical path for DeepSeek and Mimo:

```python
class OpenAIClient(BaseLM):
    def __init__(self, api_key=None, model_name=None, base_url=None, **kwargs):
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self.client = openai.OpenAI(**client_kwargs)
```

Any OpenAI-compatible API (DeepSeek, Mimo, etc.) works via:

```python
# DeepSeek
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },
)

# Mimo (if OpenAI-compatible)
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("MIMO_API_KEY"),
        "model_name": "mimo-xxx",
        "base_url": "https://api.mimo.xxx/v1",
    },
)
```

### Adding a New Custom Backend

To add a truly custom backend (not OpenAI-compatible), you'd need to:

1. Create a new client class inheriting from `BaseLM` in `rlm/clients/`
2. Implement `completion()`, `acompletion()`, `get_usage_summary()`, `get_last_usage()`
3. Register it in `get_client()` in `rlm/clients/__init__.py`

But for OpenAI-compatible APIs, **no code changes needed** — just use `backend="openai"` with a custom `base_url`.

---

## 2. `other_backends` — The Multi-Model Limitation

### Current Limitation: Only 1 Additional Backend

The constructor validates this explicitly:

```python
# rlm/core/rlm.py line 146
if other_backends is not None:
    if len(other_backends) != 1:
        raise ValueError(
            "We currently only support one additional backend for the recursive sub-calls! "
            "This model will be the model used for recursive sub-calls, but this will change in the future"
        )
```

### How `other_backends` Works

- **`other_backends[0]`** — used as the default for depth-based routing (depth=1 calls use this model)
- The first `other_backends` client is also registered by `model_name` in the LMHandler
- Additional backends beyond index 0 are registered by model name but NOT validated at init

### Workaround: Model-Name Routing via `llm_query(prompt, model="name")`

Despite the `len(other_backends) != 1` validation, there's a model-name routing mechanism in `LMHandler.get_client()`:

```python
def get_client(self, model=None, depth=0):
    if model and model in self.clients:
        return self.clients[model]        # Explicit model override
    if depth == 1 and self.other_backend_client:
        return self.other_backend_client  # Depth-based routing
    return self.default_client            # Fallback
```

And in `_subcall()`, the `model=` parameter overrides `backend_kwargs["model_name"]`:

```python
def _subcall(self, prompt, model=None):
    if model:
        child_backend_kwargs = dict(self.backend_kwargs or {})
        child_backend_kwargs["model_name"] = model
```

**The workaround:** Register additional clients manually after construction, or modify the init to skip the validation. The model-name routing in the REPL (`llm_query(prompt, model="deepseek-chat")`) will pick up any client registered by that name.

### Practical Multi-Model Setup

For a DeepSeek + Mimo setup, the cleanest approach is:

```python
# Option A: Use one as primary, one as other_backends
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": os.getenv("MIMO_API_KEY"),
        "model_name": "mimo-model",
        "base_url": "https://api.mimo.xxx/v1",
    }],
)
# Inside REPL: llm_query("...") uses Mimo (depth=1 routing)
# llm_query("...", model="deepseek-chat") uses DeepSeek explicitly
```

```python
# Option B: Use OpenRouter as a gateway (routes to both)
rlm = RLM(
    backend="openrouter",
    backend_kwargs={
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model_name": "deepseek/deepseek-chat",
    },
    other_backends=["openrouter"],
    other_backend_kwargs=[{
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model_name": "mimo/xxx",
    }],
)
```

---

## 3. REPL Environment — Available Functions

### Functions Available Inside the REPL

| Function | Signature | Description |
|----------|-----------|-------------|
| `llm_query` | `llm_query(prompt, model=None) -> str` | Single plain LM completion. Fast, no REPL, no iteration. Sub-LLM handles ~500K chars. |
| `llm_query_batched` | `llm_query_batched(prompts, model=None) -> list[str]` | Multiple plain LM completions concurrently. Failed slots return error string, rest succeed. |
| `rlm_query` | `rlm_query(prompt, model=None) -> str` | Spawn a child RLM with its own REPL for deeper thinking. Falls back to `llm_query` at max depth. |
| `rlm_query_batched` | `rlm_query_batched(prompts, model=None) -> list[str]` | Spawn multiple child RLMs. Falls back to `llm_query_batched` at max depth. |
| `SHOW_VARS` | `SHOW_VARS() -> str` | List all user-created variables in the REPL. |
| `answer` | `dict` | `{"content": "", "ready": False}` — set `answer["content"]` and `answer["ready"] = True` to finish. |
| `context` | variable | The input prompt/payload. Alias for `context_0`. |
| `print(...)` | builtin | Output visible to the model in the next iteration. |

### Model Routing Inside REPL

```python
# In REPL code:
llm_query("What is 2+2?")                           # → other_backend (depth=1) or default
llm_query("What is 2+2?", model="deepseek-chat")    # → explicitly DeepSeek
llm_query("What is 2+2?", model="mimo-model")       # → explicitly Mimo
```

### `llm_query` vs `rlm_query`

- **`llm_query`** — For simple, one-shot tasks: extracting info, summarizing, classifying. Fast single LLM call.
- **`rlm_query`** — For subtasks requiring deeper reasoning: multi-step problem solving, code execution, iterative problem-solving. The child gets its own REPL.

---

## 4. System Prompt Customization

### `custom_system_prompt`

Override the default RLM system prompt entirely:

```python
custom_prompt = """You are a software development assistant.
Use the REPL to analyze the context variable (code, docs, etc).
Available tools:
- llm_query(prompt) for quick lookups
- rlm_query(prompt) for complex sub-tasks
When done, set answer["content"] and answer["ready"] = True.
"""

rlm = RLM(..., custom_system_prompt=custom_prompt)
```

### Default System Prompt Structure

The default prompt (`RLM_SYSTEM_PROMPT` in `rlm/utils/prompts.py`) includes:
1. Context variable instructions
2. REPL usage guidance
3. `llm_query` / `rlm_query` documentation
4. Custom tools section (auto-injected from `custom_tools`)
5. Orchestrator addendum (if `orchestrator=True`)

### `orchestrator` Mode

When `orchestrator=True` (default), an addendum is appended instructing the model to act as an orchestrator — delegating work to sub-LLMs rather than solving directly.

### `user_prologue`

An optional message inserted between the metadata and the first iteration prompt:

```python
rlm = RLM(..., user_prologue="Focus on code quality and security.")
```

---

## 5. Persistent Environment

### How `persistent=True` Works

When enabled, the same environment (REPL) is reused across multiple `completion()` calls:

```python
with RLM(..., persistent=True) as rlm:
    result1 = rlm.completion("First context")    # context_0 loaded
    result2 = rlm.completion("Second context")   # context_1 loaded, can access context_0
    result3 = rlm.completion("Third context")    # context_2 loaded, can access all
```

### Key Behaviors

- **Contexts are versioned:** `context_0`, `context_1`, `context_2`, ... with `context` aliasing `context_0`
- **Histories are versioned:** `history_0`, `history_1`, ... with `history` aliasing `history_0`
- **Variables persist:** Any variable created in one `completion()` is available in subsequent calls
- **Environment reuse:** The `LocalREPL` instance (and its namespace) stays alive across calls
- **LM handler is recreated** per completion (but the environment survives)

### Protocol: `SupportsPersistence`

Environments must implement:
- `update_handler_address(address)` — update the LM handler for new completion
- `add_context(payload, index)` — add context as `context_N`
- `get_context_count()` — return number of contexts
- `add_history(history, index)` — add message history as `history_N`
- `get_history_count()` — return number of histories

### Cleanup

```python
# Context manager (recommended)
with RLM(..., persistent=True) as rlm:
    rlm.completion("...")
# Auto-cleanup on exit

# Manual
rlm = RLM(..., persistent=True)
rlm.completion("...")
rlm.close()  # Clean up persistent environment
```

---

## 6. Compaction — Auto-Summarization

### How Compaction Works

When `compaction=True`, RLM monitors token usage and auto-summarizes when the context window fills up:

```python
rlm = RLM(
    ...,
    compaction=True,
    compaction_threshold_pct=0.85,  # Trigger at 85% of context limit
)
```

### Compaction Flow

1. Before each iteration, check: `current_tokens >= threshold_tokens`
2. If triggered, the LM summarizes the conversation so far:
   - "Summarize your progress: completed steps, intermediate results, next action"
3. Summary is appended to the REPL's `history` variable
4. Message history is reset to: `[system, metadata, summary, "Continue from summary..."]`
5. The model continues from the summary, with `history` containing the full trajectory

### Compaction History in REPL

```python
# In REPL, after compaction:
print(history)  # Full trajectory segments + summaries
# history is a list containing:
# - Original message dicts (trajectory segments)
# - {"type": "summary", "content": "..."} dicts
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compaction` | `False` | Enable auto-summarization |
| `compaction_threshold_pct` | `0.85` | Fraction of context window that triggers compaction |

### Token Counting

RLM uses `get_context_limit(model_name)` and `count_tokens(messages, model_name)` to determine when to compact. These are model-aware.

---

## 7. Custom Tools — Injecting Python Functions

### Two Formats

```python
custom_tools = {
    # Plain format
    "fetch_data": my_function,

    # Dict format with description (recommended)
    "calculator": {
        "tool": calc_function,
        "description": "Performs arithmetic calculations",
    },

    # Non-callable values (data/constants)
    "API_KEY": {"tool": "sk-...", "description": "API key for services"},
}
```

### Reserved Names (Cannot Override)

`llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `context`, `history`, `answer`, `SHOW_VARS`

### Sub-Agent Tools

```python
# Separate tools for child RLMs
rlm = RLM(
    ...,
    custom_tools={"fetch_data": fetch_fn},           # For root REPL
    custom_sub_tools={"limited_tool": limited_fn},    # For child RLMs
)
# Pass {} to disable all tools for sub-agents
```

---

## 8. Integration Strategy for RLM-Server

### Architecture

```
VS Code Copilot → OpenAI-compatible API → RLM-Server → RLM Library
                                                      ├── DeepSeek (primary)
                                                      └── Mimo (other_backend)
```

### Recommended Configuration

```python
from rlm import RLM
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    # Primary model: DeepSeek
    backend="openai",
    backend_kwargs={
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },

    # Sub-call model: Mimo
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": os.getenv("MIMO_API_KEY"),
        "model_name": "mimo-model",
        "base_url": "https://api.mimo.xxx/v1",
    }],

    # REPL settings
    environment="local",
    max_depth=2,
    max_iterations=30,

    # Dev assistant system prompt
    custom_system_prompt="You are a software development assistant...",
    orchestrator=True,

    # Long conversation support
    persistent=True,
    compaction=True,
    compaction_threshold_pct=0.85,

    # Tools for code analysis
    custom_tools={
        "read_file": {"tool": read_file_fn, "description": "Read file contents"},
        "search_code": {"tool": search_fn, "description": "Search codebase"},
    },

    # Logging
    logger=logger,
    verbose=True,
)
```

### OpenAI-Compatible API Wrapper

To expose RLM as an OpenAI-compatible API for VS Code Copilot:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    # ...

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Extract user message
    user_msg = next(m["content"] for m in request.messages if m["role"] == "user")

    # Run RLM
    result = rlm.completion(user_msg)

    # Return OpenAI-compatible response
    return {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.response},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
```

---

## 9. Pitfalls & Gotchas

| Pitfall | Details |
|---------|---------|
| **`other_backends` limit** | Only 1 additional backend allowed at init. Workaround: model-name routing or monkey-patching. |
| **`model_name` required** | OpenAI client raises if `model_name` is None. Always set it. |
| **`base_url` path** | DeepSeek uses `/v1`, not `/v1/chat/completions`. The OpenAI SDK handles the path. |
| **Token counting** | Compaction uses model-specific context limits. Unknown models default to a generic limit. |
| **Persistent + compaction** | When `persistent=True` and `compaction=True`, the `history` variable is the compaction history, not the versioned histories. |
| **REPL output truncation** | REPL outputs over ~20K chars are truncated. Use `llm_query` on large variables instead of `print`. |
| **Local REPL isolation** | Uses `_SAFE_BUILTINS` — `eval`, `exec`, `input` are removed from builtins. |
| **Cost tracking** | OpenAI-compatible APIs may not return cost data. Budget features rely on provider reporting. |
