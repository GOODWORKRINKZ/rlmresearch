# Phase 1: RLM Core + Basic API - Context

**Gathered:** 2026-06-14
**Status:** ✅ Completed 2026-06-14
**Plans:** 01-01 ✅ | 01-02 ✅ | 01-03 ✅

<domain>
## Phase Boundary

This phase delivers a working RLM server that:
1. Uses DeepSeek API as the LLM backend
2. Exposes OpenAI-compatible `/v1/chat/completions` endpoint via FastAPI
3. Can be connected to VS Code Copilot as a custom endpoint

NOT in this phase: multi-model routing, custom tools, RAG, Mimo integration (those are Phase 2-3).

</domain>

<decisions>
## Implementation Decisions

### RLM Library
- Use official `rlms` library (pip install rlms, github.com/alexzhang13/rlm)
- Backend: `backend="openai"` with `base_url` pointing to DeepSeek API
- DeepSeek API is OpenAI-compatible, so no custom backend code needed
- Use `custom_system_prompt` for dev assistant behavior

### DeepSeek API Integration
- Base URL: https://api.deepseek.com
- Models: `deepseek-v4-flash` (cheap, fast) and `deepseek-v4-pro` (quality)
- Phase 1 uses `deepseek-v4-flash` for all calls (cheaper for initial development)
- Authentication: API key via `Authorization: Bearer <key>` header
- Context caching is automatic (no code needed)

### FastAPI Server
- Endpoint: `/v1/chat/completions` (OpenAI-compatible)
- Request format: standard OpenAI chat completions format
- Response format: standard OpenAI chat completions format
- Streaming: SSE (Server-Sent Events) for real-time responses
- Port: configurable, default 8000

### VS Code Copilot Integration
- Configure as custom endpoint in VS Code settings
- No proxy needed — direct connection to FastAPI server
- Model name in requests: can be any string (RLM handles routing)

### Configuration
- API keys via environment variables (DEEPSEEK_API_KEY)
- Server config via config file or env vars
- RLM parameters: verbose=True, persistent=True for dev work

### the agent's Discretion
- Exact project structure (src layout vs flat)
- Logging format and level
- Error handling specifics
- Health check endpoint implementation
- CORS configuration for VS Code

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### RLM Library
- `.planning/research/rlm-integration.md` — RLM library internals, REPL functions, custom_tools, system prompt

### API Integration
- `.planning/research/api-integration.md` — DeepSeek API details, pricing, model names, OpenAI-compatible format

### Project Definition
- `.planning/PROJECT.md` — Project vision, requirements, key decisions
- `.planning/ROADMAP.md` — Phase goals and verification criteria

</canonical_refs>

<specifics>
## Specific Ideas

### Quick Start Code Pattern
```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
    custom_system_prompt="You are a software development assistant...",
    verbose=True,
    persistent=True,
)

# Use in FastAPI endpoint
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    response = rlm.completion(request.messages[-1].content)
    return {"choices": [{"message": {"content": response.response}}]}
```

### System Prompt for Dev Assistant
```
You are an expert software development assistant powered by Recursive Language Models (RLM).
You help developers with:
- Code analysis and understanding
- Bug finding and fixing
- Code generation and refactoring
- Documentation writing
- Architecture decisions

When given a complex task, decompose it into subtasks and process them recursively.
Always provide concrete, actionable code examples.
```

### VS Code Copilot Configuration
```json
{
  "github.copilot.advanced": {
    "customEndpoint": {
      "url": "http://localhost:8000/v1/chat/completions",
      "model": "rlm-dev-assistant"
    }
  }
}
```

</specifics>

<deferred>
## Deferred Ideas

- Multi-model routing (Phase 2)
- Custom tools for code analysis (Phase 2)
- RAG for codebase indexing (Phase 3)
- Mimo integration (Phase 2)
- Production optimizations (Phase 4)

</deferred>

---

*Phase: 01-rlm-core-api*
*Context gathered: 2026-06-14*
