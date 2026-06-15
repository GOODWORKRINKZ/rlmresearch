# Plan 02-03 SUMMARY: System Prompt + VS Code Tool Calling Fix

**Status:** ✅ COMPLETED
**Executed:** 2026-06-15
**Commit:** `feat(02-03): system prompt with tool docs + VS Code tool-stripping`

## What Was Built

Updated system prompt with tool documentation and multi-model routing guidance. Fixed VS Code tool calling compatibility by stripping OpenAI-style tool definitions before passing to RLM. Added orchestrator flag for multi-model delegation behavior.

## Changes Made

### src/rlm_assistant/system_prompt.py
- Added "Available Tools" section documenting all 4 custom tools with usage examples
- Added "Multi-Model Routing" section explaining Pro/Flash depth-based routing
- Updated Guidelines to prefer custom tools over repl code for I/O

### src/rlm_assistant/server.py
- `ChatCompletionRequest` now accepts `tools: Optional[list]` and `tool_choice: Optional[str]` fields
- Added `sanitize_messages()` function that strips `role="tool"` messages
- Updated `chat_completions` handler to sanitize messages and log tool stripping
- Updated health endpoint to show pro_model and flash_model
- Updated startup log with new model field names

### src/rlm_assistant/config.py
- Added `rlm_orchestrator: bool` field (default True, from `RLM_ORCHESTRATOR` env var)

### src/rlm_assistant/rlm_client.py
- Wired `orchestrator=settings.rlm_orchestrator` into RLM constructor

## Verification

All checks passed:
- `DEV_USER_PROLOGUE` contains all 4 tool names, `llm_query`, `rlm_query`, `answer`, `Pro`
- `ChatCompletionRequest` accepts `tools` and `tool_choice` fields
- `sanitize_messages()` correctly strips tool-role messages
- `Settings.rlm_orchestrator` defaults to `True`
- `create_rlm()` includes `orchestrator` parameter

## VS Code Compatibility

When VS Code sends requests with `tools: [...]` and `tool_choice: "auto"`:
1. Fields are parsed by Pydantic (no 422 error)
2. Warning logged: "Stripping N tool definitions from VS Code request"
3. Tool-role messages filtered out by `sanitize_messages()`
4. Only user/system/assistant messages passed to RLM
