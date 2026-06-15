# Plan 02-01 SUMMARY: Model Routing (Pro for root, Flash for sub-calls)

**Status:** ✅ COMPLETED
**Executed:** 2026-06-15
**Commit:** `feat(02-01): multi-model routing — Pro for root, Flash for sub-calls`

## What Was Built

Multi-model routing configuration: DeepSeek V4 Pro for root-level complex reasoning, DeepSeek V4 Flash for sub-calls (llm_query, rlm_query at depth > 0). Automatic depth-based routing saves cost without sacrificing quality.

## Changes Made

### src/rlm_assistant/config.py
- Replaced `deepseek_model` field with `deepseek_pro_model` + `deepseek_flash_model`
- Pro model loaded from `DEEPSEEK_PRO_MODEL` env var (default: `deepseek-v4-pro`)
- Flash model loaded from `DEEPSEEK_FLASH_MODEL` env var (default: `deepseek-v4-flash`)
- Backward compat: `DEEPSEEK_MODEL` env var works as pro_model fallback
- Added `other_backend_kwargs` property returning Flash config as list[dict]
- Updated `rlm_backend_kwargs` to use Pro model
- Updated logging to show both model names

### src/rlm_assistant/rlm_client.py
- Wired `other_backends=["openai"]` into RLM constructor
- Wired `other_backend_kwargs=settings.other_backend_kwargs` into RLM constructor
- Updated logging to show pro_model and flash_model names
- Updated module docstring to document multi-model routing

### .env.example
- Updated with new `DEEPSEEK_PRO_MODEL` and `DEEPSEEK_FLASH_MODEL` variable names

## Verification

All automated checks passed:
- `Settings.deepseek_pro_model` == `deepseek-v4-pro`
- `Settings.deepseek_flash_model` == `deepseek-v4-flash`
- `rlm_backend_kwargs` uses Pro model
- `other_backend_kwargs` returns Flash config
- `create_rlm()` includes `other_backends` and `other_backend_kwargs` in source

## Routing Behavior

- **depth=0 (root):** DeepSeek V4 Pro — complex reasoning, planning, orchestration
- **depth=1 (sub-calls):** DeepSeek V4 Flash — fast, cheap, good for simple lookups
- **Explicit override:** `llm_query(prompt, model="deepseek-v4-pro")` forces Pro at any depth
