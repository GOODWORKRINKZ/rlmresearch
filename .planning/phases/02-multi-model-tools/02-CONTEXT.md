# Phase 2: Multi-Model + Custom Tools - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers:
1. Multi-model routing — DeepSeek Pro for complex tasks, Flash for sub-calls
2. Custom tools inside RLM REPL — file operations, command execution, code analysis
3. System prompt tuning — orchestrator for multi-model delegation
4. Mimo integration — alternative backend

NOT in this phase: RAG, vector DB, code indexing (Phase 3), production optimization (Phase 4).

## Current State (from Phase 1)

- RLM server works with DeepSeek V4 Flash on port 8000
- OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming
- `user_prologue` used instead of `custom_system_prompt` (preserves RLM's ```repl instructions)
- Server registered in VS Code as custom endpoint (`toolCalling: true`)
- **Problem discovered**: RLM doesn't support OpenAI-style tool calling — VS Code sends tool definitions as context, model gets confused

</domain>

<decisions>
## Implementation Decisions

### Multi-Model Routing (P2.1)
- DeepSeek V4 Pro (`deepseek-v4-pro`) for root-level complex reasoning
- DeepSeek V4 Flash (`deepseek-v4-flash`) for sub-calls (llm_query, rlm_query at depth > 0)
- Use RLM's `other_backends` parameter for secondary model
- Inside REPL: `llm_query(prompt)` uses Flash, `llm_query(prompt, model="deepseek-v4-pro")` uses Pro explicitly

### Custom Tools (P2.2)
- RLM's `custom_tools` dict injects Python functions into REPL namespace
- Tools available: `read_file(path)`, `write_file(path, content)`, `run_command(cmd)`, `search_code(pattern, path)`
- Reserved names cannot be overridden: `llm_query`, `rlm_query`, `context`, `answer`, `SHOW_VARS`
- Use dict format with descriptions: `{"tool": fn, "description": "..."}`
- `custom_sub_tools` for child RLMs (more restricted)

### System Prompt (P2.3)
- Keep `user_prologue` approach (proven in Phase 1)
- Add tool documentation to prologue
- `orchestrator=True` for multi-model delegation behavior

### Mimo Integration (P2.4)
- Mimo as alternative backend via `backend="openai"` + custom `base_url`
- Config-driven: switch models via .env variables
- Fallback: if DeepSeek fails, try Mimo

### the agent's Discretion
- Exact tool function signatures and error handling
- Tool timeout limits
- Logging format for tool calls
- How to handle tool errors in REPL

</decisions>

<canonical_refs>
## Canonical References

### RLM Library
- `.planning/research/rlm-integration.md` — RLM library internals, REPL functions, custom_tools, system prompt

### API Integration
- `.planning/research/api-integration.md` — DeepSeek API details, pricing, model names

### Phase 1 Code
- `src/rlm_assistant/config.py` — Settings with rlm_backend_kwargs
- `src/rlm_assistant/rlm_client.py` — create_rlm() with user_prologue
- `src/rlm_assistant/server.py` — FastAPI server with streaming
- `src/rlm_assistant/system_prompt.py` — DEV_USER_PROLOGUE

### Project Definition
- `.planning/PROJECT.md` — Project vision
- `.planning/ROADMAP.md` — Phase goals and verification criteria

</canonical_refs>

<specifics>
## Key Technical Details

### RLM custom_tools Pattern
```python
custom_tools = {
    "read_file": {
        "tool": lambda path: open(path).read(),
        "description": "Read file contents",
    },
    "run_command": {
        "tool": lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout,
        "description": "Run shell command",
    },
}
rlm = RLM(..., custom_tools=custom_tools)
```

### Multi-Model via other_backends
```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "deepseek-v4-pro", ...},
    other_backends=["openai"],
    other_backend_kwargs=[{
        "model_name": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "...",
    }],
)
# In REPL: llm_query("...") uses Flash (depth=1)
# llm_query("...", model="deepseek-v4-pro") uses Pro
```

### VS Code Tool Calling Problem
- VS Code sends `tools: [...]` in request body when `toolCalling: true`
- RLM doesn't understand OpenAI function calling format
- Solution: Either strip tool definitions from request before passing to RLM, or set `toolCalling: false` and accept no agent mode

</specifics>
