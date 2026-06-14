---
phase: 1
reviewers: [deepseek-v4-pro, deepseek-v4-flash]
reviewed_at: 2026-06-14T20:30:00Z
plans_reviewed: [01-01-PLAN.md, 01-02-PLAN.md, 01-03-PLAN.md]
---

# Cross-AI Plan Review — Phase 1

## DeepSeek V4 Pro Review

Below is structured feedback on the three implementation plans for Phase 1.

### Plan 01-01 — Project scaffold & config

**Summary** — Correctly sets up a modern Python project with pyproject.toml, environment variable loading, and a typed configuration module.

**Strengths**
- Clear artifact list and verifiable acceptance criteria
- Uses python-dotenv for .env auto-loading
- Configuration collected in single module; rlm_backend_kwargs property avoids scattered dicts

**Concerns**
- (LOW) Entry point references server:main which doesn't exist yet — safe but will cause import error if run early
- (LOW) No validation that deepseek_model matches a known value

**Risk: LOW**

### Plan 01-02 — RLM client wrapper & system prompt

**Summary** — Wraps RLM library with DeepSeek backend and defines dev-focused system prompt. Interface (chat(message: str)) is too primitive for OpenAI-compatible server.

**Strengths**
- System prompt correctly references llm_query, rlm_query, and answer dict
- Singleton get_rlm() avoids recreating REPL environment
- Uses logging instead of print

**Concerns**
- **(HIGH) chat(message: str) discards multi-turn conversation state** — VS Code Copilot sends full message history; extracting only last user message breaks multi-turn interaction
- (MEDIUM) RLM's completion() called with plain string — system prompt context may be lost
- (LOW) No handling of DeepSeek's reasoning_content tokens

**Risk: HIGH**

### Plan 01-03 — FastAPI server

**Summary** — Exposes OpenAI-compatible endpoints with Pydantic models, CORS, health, and model listing. Inherits multi-turn weakness from plan 01-02.

**Strengths**
- Pydantic models faithfully replicate OpenAI schema
- /v1/models and /health meet phase requirements
- CORS correct for local dev
- Self-contained TestClient test snippet

**Concerns**
- **(HIGH) POST /v1/chat/completions extracts only last user message** — fails "VS Code Copilot интеграция" requirement
- (MEDIUM) Error handling mentioned but not implemented — should return OpenAI-compatible error JSON
- (LOW) No request validation beyond Pydantic (empty messages list)

**Risk: HIGH**

### Overall Risk: HIGH

Single-turn assumption in plans 01-02 and 01-03 directly contradicts "VS Code Copilot интеграция" requirement. Foundation (01-01) is solid, but interface gap must be addressed.

---

## DeepSeek V4 Flash Review

### Plan 01-01: Project Scaffold

**Summary** — Creates project skeleton with correct dependencies and configuration model.

**Strengths**
- Correct dependency choices (rlms, fastapi, uvicorn, openai, python-dotenv, pydantic)
- Pydantic-based settings with validation and rlm_backend_kwargs property
- get_settings() cached, idempotent design
- Early validation for missing DEEPSEEK_API_KEY

**Concerns**
- (MEDIUM) pyproject.toml declares [project.scripts] before server.py exists — safe but confusing
- (LOW) No CI or linting setup — out of scope

**Risk: LOW**

### Plan 01-02: RLM Client + System Prompt

**Summary** — Creates system_prompt.py and rlm_client.py with singleton pattern. Correct RLM constructor parameters.

**Strengths**
- System prompt quality matches RLM REPL capabilities
- Singleton get_rlm() crucial for persistent=True
- RLM constructor parameters correct

**Concerns**
- (MEDIUM) System prompt mentions rlm_query() but Phase 1 has no other_backends — may spawn costly recursive calls on Flash
- (MEDIUM) chat() error handling not specified — should raise exception for server to catch
- (LOW) Singleton get_rlm() not used inside chat() default

**Risk: LOW**

### Plan 01-03: FastAPI Server

**Summary** — Builds server with three endpoints, correct OpenAI format, CORS, startup initialization.

**Strengths**
- OpenAI-compatible format
- CORS enabled for VS Code Copilot
- Early RLM initialization on startup
- Streaming deferred to Phase 4

**Concerns**
- **(MEDIUM) Conversation history not passed to RLM** — server extracts only last user message; for Copilot which sends full history, conversation continuity is broken
- (MEDIUM) Error handling missing — should return OpenAI-compatible error JSON
- (LOW) No validation that messages list is non-empty
- (LOW) Token usage hardcoded as 0

**Risk: MEDIUM**

### Overall Risk: MEDIUM

Primary risk: conversation history not passed to RLM. If resolved (by concatenating messages), risk drops to LOW.

---

## Consensus Summary

### Agreed Strengths (both reviewers)
- Plan 01-01 is solid — correct dependencies, configuration model, good ergonomics
- System prompt quality is good and matches RLM capabilities
- Singleton pattern for RLM client is correct design choice
- Scope discipline — no over-engineering, clear phase boundaries
- Automated verification snippets are valuable

### Agreed Concerns (both reviewers)

| Severity | Concern | Plans |
|----------|---------|-------|
| **HIGH** | Conversation history not passed to RLM — only last user message extracted, breaks VS Code Copilot multi-turn | 01-02, 01-03 |
| **MEDIUM** | Error handling not implemented — should return OpenAI-compatible error JSON | 01-02, 01-03 |
| **MEDIUM** | chat() error behavior unspecified (raise vs return None) | 01-02 |
| **LOW** | No validation for empty messages list | 01-03 |

### Divergent Views
- **Risk level**: Pro rates overall HIGH, Flash rates MEDIUM — difference is in severity assessment of conversation history gap
- **Reasoning tokens**: Pro flags DeepSeek reasoning_content handling; Flash doesn't mention it
- **Recursive calls**: Flash notes system prompt may trigger costly rlm_query on Flash model; Pro doesn't flag this

### Recommended Fix
Before execution, modify Plan 01-03 to **concatenate all messages into a single prompt**:
```python
# Instead of extracting last user message only:
full_prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
response = chat(full_prompt)
```
This preserves conversation context for Copilot without over-engineering.
