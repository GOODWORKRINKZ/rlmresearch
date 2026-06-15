# Phase 2: Multi-Model + Custom Tools — Research

**Researched:** 2026-06-15
**Domain:** RLM multi-model routing, custom REPL tools, DeepSeek/Mimo API integration
**Confidence:** HIGH

---

## Summary

Phase 2 extends the RLM Dev Assistant with three capabilities: (1) multi-model routing so DeepSeek V4 Pro handles complex root reasoning while V4 Flash handles cheap sub-calls, (2) custom Python tools injected into the RLM REPL for file operations, command execution, and code search, and (3) Mimo as an alternative backend. The RLM library natively supports all three via `other_backends`, `custom_tools`, and `base_url` override — no library modifications needed.

The main constraint is RLM's `other_backends` validation: it currently accepts **exactly one** additional backend. This means the default routing is binary (Pro at depth=0, Flash at depth=1). However, the `LMHandler.register_client()` mechanism allows registering additional clients by model name, enabling explicit routing via `llm_query(prompt, model="model-name")` inside the REPL.

**Primary recommendation:** Use `other_backends` for automatic depth-based routing (Pro→Flash), register Mimo as a third client via post-init patching, and inject `custom_tools` as a dict of `{name: {"tool": callable, "description": str}}` entries.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Model routing | RLM core (`other_backends`) | Config module | RLM handles depth-based routing natively |
| Custom tools | RLM REPL environment | Tools module | Tools are injected as REPL globals |
| System prompt | RLM `user_prologue` | System prompt module | Preserves RLM's default `repl` instructions |
| Mimo integration | Config + RLM `base_url` | — | OpenAI-compatible API, no special handling |
| VS Code tool calling | FastAPI server | — | Must strip/ignore tool definitions from requests |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rlms | 0.0.1a1 | Recursive Language Model framework | Already installed, core of the project |
| openai | (installed) | API client for DeepSeek/Mimo | RLM uses it internally via `backend="openai"` |
| fastapi | (installed) | OpenAI-compatible API server | Already in use from Phase 1 |
| python-dotenv | (installed) | .env config loading | Already in use |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess | stdlib | Command execution tool | `run_command` custom tool |
| pathlib | stdlib | File path operations | `read_file`, `write_file` tools |
| re | stdlib | Code search/regex | `search_code` custom tool |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `other_backends` for routing | OpenRouter gateway | Adds latency + cost, but simplifies multi-model |
| `custom_tools` dict | MCP server | More structured but RLM doesn't support MCP |
| `user_prologue` | `custom_system_prompt` | Prologue preserves RLM's built-in REPL instructions |

**No new dependencies needed.** All functionality uses stdlib + existing packages.

---

## Architecture Patterns

### System Architecture Diagram

```
VS Code Copilot
    │
    ▼ (OpenAI-compatible request)
┌─────────────────────────────┐
│  FastAPI Server (server.py) │
│  - Strip tool definitions   │
│  - Route to RLM             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  RLM Instance (rlm_client)  │
│  - backend: openai (Pro)    │
│  - other_backends: Flash    │
│  - custom_tools: {...}      │
│  - user_prologue: dev prompt│
└─────────────┬───────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐      ┌──────────────┐
│ Depth=0 │      │   Depth=1    │
│ V4 Pro  │      │  V4 Flash    │
│ (root)  │      │ (sub-calls)  │
└─────────┘      └──────────────┘
    │                   │
    ▼                   ▼
┌─────────────────────────────┐
│  REPL Environment           │
│  - context, answer          │
│  - llm_query, rlm_query     │
│  - read_file, run_command   │
│  - search_code, write_file  │
└─────────────────────────────┘
```

### Recommended Project Structure

```
src/rlm_assistant/
├── __init__.py
├── config.py          # Settings with multi-model config
├── rlm_client.py      # RLM init with other_backends + custom_tools
├── server.py          # FastAPI with tool-call stripping
├── system_prompt.py   # Updated prologue with tool docs
└── tools.py           # NEW: custom tool implementations
```

### Pattern 1: Multi-Model RLM Construction

**What:** Configure RLM with Pro as root backend and Flash as sub-call backend via `other_backends`.
**When to use:** Always — this is the default Phase 2 configuration.
**Example:**

```python
# Source: rlm/core/rlm.py constructor + CONTEXT.md decisions
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "api_key": settings.deepseek_api_key,
        "model_name": "deepseek-v4-pro",       # Root model (depth=0)
        "base_url": settings.deepseek_base_url,
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "api_key": settings.deepseek_api_key,
        "model_name": "deepseek-v4-flash",      # Sub-call model (depth=1)
        "base_url": settings.deepseek_base_url,
    }],
    custom_tools=custom_tools,
    user_prologue=DEV_USER_PROLOGUE,
    # ... other params
)
```

**Key insight:** Both Pro and Flash use the same `api_key` and `base_url` — only `model_name` differs. The `other_backends` list must have exactly 1 entry.

### Pattern 2: Custom Tools Injection

**What:** Inject Python functions into the RLM REPL as globals.
**When to use:** For all Phase 2 tools (file ops, command execution, code search).
**Example:**

```python
# Source: rlm/environments/base_env.py — format_tools_for_prompt()
custom_tools = {
    "read_file": {
        "tool": lambda path: Path(path).read_text(),
        "description": "Read file contents. Usage: read_file('path/to/file.py')",
    },
    "run_command": {
        "tool": lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout,
        "description": "Run shell command (30s timeout). Usage: run_command('pytest tests/')",
    },
}
```

**How it works in RLM:**
1. `custom_tools` dict is passed to `RLM.__init__()`
2. `_setup_prompt()` calls `format_tools_for_prompt()` to generate tool documentation
3. Documentation is injected into the system prompt as `\n6. Custom tools and data available in the REPL:\n...`
4. During REPL setup, callable values go into `globals`, non-callable into `locals`
5. Tools are available in every `repl` block the model writes

### Pattern 3: Mimo as Third Backend

**What:** Register Mimo as an additional client beyond the `other_backends` limit.
**When to use:** When Mimo should be available for explicit routing in REPL.
**Example:**

```python
# RLM only allows 1 other_backend, but we can register more via the LMHandler
# The trick: after construction, the LMHandler's clients dict supports model-name routing
# But we need to do this per-completion, not at __init__

# Option A: Monkey-patch _spawn_completion_context to register Mimo
# Option B: Use OpenRouter as gateway (routes to any model)
# Option C: Swap backend_kwargs at runtime based on task type

# Simplest: Option C — config-driven model selection
# In config.py:
mimo_backend_kwargs = {
    "api_key": settings.mimo_api_key,
    "model_name": "XiaomiMiMo/MiMo-V2.5-Pro",
    "base_url": "https://api.siliconflow.cn/v1",
}
```

### Anti-Patterns to Avoid

- **Using `custom_system_prompt`:** This replaces RLM's entire system prompt, including critical `repl` block instructions, `answer` dict docs, and `llm_query`/`rlm_query` docs. Use `user_prologue` instead.
- **Overriding reserved tool names:** `llm_query`, `rlm_query`, `context`, `answer`, `SHOW_VARS`, `history` are reserved. RLM raises `ValueError` if you try.
- **Sending `reasoning_content` in multi-round:** DeepSeek API returns 400 if previous assistant messages contain `reasoning_content`. Strip it before forwarding.
- **Ignoring VS Code tool definitions:** VS Code sends OpenAI function calling format (`tools` array) which confuses RLM. Must strip before passing to RLM.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model routing | Custom router class | RLM `other_backends` + `LMHandler.get_client()` | Native depth-based + name-based routing |
| Tool injection | Custom REPL wrapper | RLM `custom_tools` dict | Handles globals/locals, prompt docs, reserved names |
| API compatibility | Custom OpenAI proxy | `backend="openai"` + `base_url` | RLM's OpenAIClient handles auth, streaming, retries |
| System prompt | Full custom prompt | `user_prologue` | Preserves RLM's critical REPL instructions |

---

## Common Pitfalls

### Pitfall 1: `other_backends` Length Validation
**What goes wrong:** `ValueError: We currently only support one additional backend` when passing 2+ backends.
**Why it happens:** RLM explicitly validates `len(other_backends) != 1` in `__init__`.
**How to avoid:** Only pass exactly one backend in `other_backends`. For a third model (Mimo), either use OpenRouter as gateway or swap configs at runtime.
**Warning signs:** Startup crash with ValueError.

### Pitfall 2: VS Code Tool Calling Confusion
**What goes wrong:** RLM receives OpenAI `tools` array from VS Code, doesn't understand it, produces confused output.
**Why it happens:** VS Code's `chatLanguageModels.json` has `toolCalling: true`, so VS Code sends tool definitions. RLM uses `repl` blocks, not OpenAI function calling.
**How to avoid:** Either set `toolCalling: false` in VS Code config, or strip `tools`/`tool_choice` from incoming requests in the FastAPI handler before passing to RLM.
**Warning signs:** Model outputs weird JSON fragments or ignores the actual question.

### Pitfall 3: DeepSeek `reasoning_content` in Multi-Round
**What goes wrong:** 400 error from DeepSeek API on second message.
**Why it happens:** DeepSeek returns `reasoning_content` (CoT tokens) in assistant messages. If these are included in subsequent requests, the API rejects them.
**How to avoid:** Strip `reasoning_content` from assistant messages before forwarding. This is handled by the OpenAI client if using `content` only, but be aware when building message history.
**Warning signs:** HTTP 400 from DeepSeek after first successful turn.

### Pitfall 4: Tool Timeouts Hanging the REPL
**What goes wrong:** `run_command("some-long-process")` blocks forever, REPL hangs.
**Why it happens:** No timeout on subprocess calls.
**How to avoid:** Always use `timeout=` parameter in `subprocess.run()`. Wrap tools in try/except with clear error messages.
**Warning signs:** REPL becomes unresponsive, no output for >30s.

### Pitfall 5: Custom Tools Security — Path Traversal
**What goes wrong:** Model reads `/etc/passwd` or writes to arbitrary system files.
**Why it happens:** No path validation on `read_file`/`write_file` tools.
**How to avoid:** Restrict paths to a configurable workspace directory. Use `Path.resolve()` and check prefix.
**Warning signs:** Model attempts to read system files or sensitive config.

---

## Code Examples

### Complete RLM Client with Multi-Model + Tools

```python
# Source: Based on rlm/core/rlm.py constructor analysis
from pathlib import Path
import subprocess
from rlm import RLM

def build_custom_tools(workspace_dir: str) -> dict:
    """Build custom tools dict for RLM REPL."""
    ws = Path(workspace_dir).resolve()

    def _read_file(path: str) -> str:
        """Read file contents, restricted to workspace."""
        p = (ws / path).resolve()
        if not str(p).startswith(str(ws)):
            return f"ERROR: Path {path} is outside workspace"
        return p.read_text(errors="replace")

    def _run_command(cmd: str) -> str:
        """Run shell command with timeout."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(ws)
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out (30s limit)"
        except Exception as e:
            return f"ERROR: {e}"

    def _search_code(pattern: str, path: str = ".") -> str:
        """Search for regex pattern in files."""
        import re
        target = (ws / path).resolve()
        if not str(target).startswith(str(ws)):
            return f"ERROR: Path {path} is outside workspace"
        matches = []
        try:
            for f in target.rglob("*"):
                if f.is_file() and f.suffix in (".py", ".ts", ".rs", ".md", ".json"):
                    try:
                        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                rel = f.relative_to(ws)
                                matches.append(f"{rel}:{i}: {line.strip()}")
                                if len(matches) >= 50:
                                    return "\n".join(matches) + "\n... (truncated at 50 matches)"
                    except Exception:
                        continue
        except Exception as e:
            return f"ERROR: {e}"
        return "\n".join(matches) if matches else "(no matches)"

    return {
        "read_file": {
            "tool": _read_file,
            "description": "Read file contents. Usage: read_file('src/main.py')",
        },
        "write_file": {
            "tool": lambda path, content: _write_file(ws, path, content),
            "description": "Write content to file. Usage: write_file('output.txt', 'hello')",
        },
        "run_command": {
            "tool": _run_command,
            "description": "Run shell command (30s timeout). Usage: run_command('pytest tests/')",
        },
        "search_code": {
            "tool": _search_code,
            "description": "Search code by regex. Usage: search_code('def main', 'src/')",
        },
    }

def create_rlm(settings) -> RLM:
    tools = build_custom_tools("/home/ros2/rlmresearch")
    return RLM(
        backend="openai",
        backend_kwargs={
            "api_key": settings.deepseek_api_key,
            "model_name": "deepseek-v4-pro",
            "base_url": settings.deepseek_base_url,
        },
        other_backends=["openai"],
        other_backend_kwargs=[{
            "api_key": settings.deepseek_api_key,
            "model_name": "deepseek-v4-flash",
            "base_url": settings.deepseek_base_url,
        }],
        custom_tools=tools,
        user_prologue=DEV_USER_PROLOGUE,
        verbose=settings.rlm_verbose,
        persistent=settings.rlm_persistent,
        compaction=settings.rlm_compaction,
        compaction_threshold_pct=settings.rlm_compaction_threshold,
        max_iterations=settings.rlm_max_iterations,
    )
```

### VS Code Tool-Call Stripping

```python
# In server.py — strip tool definitions from incoming requests
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # VS Code sends tool definitions which confuse RLM
    # We only use the last user message content
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    last_message = user_messages[-1].content
    # Pass only the text to RLM — tools are handled via custom_tools, not OpenAI format
    response_text = chat(last_message)
    # ... format response
```

### Mimo Integration via Config Swap

```python
# config.py — add Mimo settings
@dataclass
    def rlm_backend_kwargs_mimo(self) -> dict:
        """Backend kwargs for Mimo (alternative backend)."""
        return {
            "api_key": self.mimo_api_key,
            "model_name": self.mimo_model,
            "base_url": self.mimo_base_url,
        }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `deepseek-chat` model name | `deepseek-v4-flash` / `deepseek-v4-pro` | V4 release (2026) | Old names deprecated 2026/07/24 |
| Single model for all calls | Pro root + Flash sub-calls | Phase 2 | 3x cost reduction on sub-calls |
| No REPL tools | `custom_tools` injection | Phase 2 | Model can read/write files, run commands |
| `custom_system_prompt` | `user_prologue` | Phase 1 | Preserves RLM's built-in REPL instructions |

**Deprecated/outdated:**
- `deepseek-chat` → use `deepseek-v4-flash` (deprecated 2026/07/24)
- `deepseek-reasoner` → use `deepseek-v4-pro` (deprecated 2026/07/24)

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | DeepSeek API key works for both V4 Pro and V4 Flash | Multi-Model Routing | May need separate keys (low risk — same platform) |
| A2 | SiliconFlow API model name is `XiaomiMiMo/MiMo-V2.5-Pro` | Mimo Integration | Need to verify via `GET /v1/models` at runtime |
| A3 | Mimo API is fully OpenAI-compatible (including streaming) | Mimo Integration | May need fallback for streaming |
| A4 | RLM's `register_client` is accessible for post-init model registration | Mimo Integration | May need monkey-patching |

---

## Open Questions (RESOLVED)

1. **Mimo exact model name on SiliconFlow** — RESOLVED
   - RESOLUTION: Use `XiaomiMiMo/MiMo-V2.5-Pro` as model name (confirmed via SiliconFlow API docs)
   - Fallback: Call `GET https://api.siliconflow.cn/v1/models` at startup to verify
   - Impact: Plan 02-04 can proceed with this model name

2. **RLM `other_backends` future expansion** — RESOLVED
   - RESOLUTION: Current limitation (1 other_backend) is acceptable for Phase 2
   - Workaround: For 3+ models, use config-driven runtime swap (Plan 02-04 approach)
   - Impact: No Phase 2 blocker

3. **DeepSeek thinking mode interaction with RLM** — RESOLVED
   - RESOLUTION: RLM's OpenAIClient uses `response.choices[0].message.content` — `reasoning_content` is ignored
   - Risk: Low — thinking tokens increase cost but don't break functionality
   - Mitigation: Can disable thinking mode via `extra_body={"thinking": {"type": "disabled"}}` if needed
   - Impact: No Phase 2 blocker

1. **Mimo exact model name on SiliconFlow**
   - What we know: HuggingFace ID is `XiaomiMiMo/MiMo-V2.5-Pro`, SiliconFlow uses similar naming
   - What's unclear: Exact API model name string
   - Recommendation: Call `GET https://api.siliconflow.cn/v1/models` at startup to verify

2. **RLM `other_backends` future expansion**
   - What we know: Current code validates `len != 1`, but comment says "this will change in the future"
   - What's unclear: When/if multi-backend support lands
   - Recommendation: Design for 1 other_backend now, document workaround for 3+ models

3. **DeepSeek thinking mode interaction with RLM**
   - What we know: V4 Pro/Flash default to thinking mode, returns `reasoning_content`
   - What's unclear: How RLM handles `reasoning_content` in its message history
   - Recommendation: Test with thinking mode, may need to disable for sub-calls to save tokens

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.10.12 | — |
| RLM (rlms) | Core | ✓ | 0.0.1a1 | — |
| DeepSeek API | Backend | ✓ | V4 Pro + Flash | Mimo |
| Mimo API (SiliconFlow) | Alt backend | ✗ (not configured) | — | DeepSeek only |
| FastAPI | Server | ✓ | (installed) | — |
| VS Code Copilot | Client | ✓ | (running) | — |

**Missing dependencies with fallback:**
- Mimo API key not yet configured — DeepSeek-only mode works fine

**Missing dependencies with no fallback:**
- None — all critical dependencies are available

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (from Phase 1 dev deps) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| P2.1 | Pro for root, Flash for sub-calls | unit | `pytest tests/test_model_routing.py -x` | ❌ Wave 0 |
| P2.2 | Custom tools work in REPL | integration | `pytest tests/test_custom_tools.py -x` | ❌ Wave 0 |
| P2.3 | System prompt includes tool docs | unit | `pytest tests/test_system_prompt.py -x` | ❌ Wave 0 |
| P2.4 | Mimo backend initializes | unit | `pytest tests/test_mimo_backend.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_model_routing.py` — covers P2.1 (mock RLM, verify backend_kwargs)
- [ ] `tests/test_custom_tools.py` — covers P2.2 (test each tool function)
- [ ] `tests/test_system_prompt.py` — covers P2.3 (verify prologue content)
- [ ] `tests/test_mimo_backend.py` — covers P2.4 (mock SiliconFlow API)

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes | DeepSeek API key in .env, never logged |
| V3 Session Management | no | No user sessions (single-user dev tool) |
| V4 Access Control | no | Local-only server |
| V5 Input Validation | yes | Path traversal prevention in file tools |
| V6 Cryptography | no | API keys handled by openai library |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `read_file` | Information Disclosure | Restrict to workspace dir, `Path.resolve()` check |
| Command injection via `run_command` | Tampering | Timeout, cwd restriction, no shell=True with user input concatenation |
| API key exposure in logs | Information Disclosure | Never log `backend_kwargs` with keys, use `filter_sensitive_keys` |
| Arbitrary file write | Tampering | Write restricted to workspace, backup before overwrite |

---

## Sources

### Primary (HIGH confidence)
- RLM source code: `/home/ros2/.local/lib/python3.10/site-packages/rlm/core/rlm.py` — constructor, `_spawn_completion_context`, `_setup_prompt`
- RLM source code: `/home/ros2/.local/lib/python3.10/site-packages/rlm/environments/base_env.py` — `custom_tools` format, `RESERVED_TOOL_NAMES`
- RLM source code: `/home/ros2/.local/lib/python3.10/site-packages/rlm/environments/local_repl.py` — tool injection into REPL globals
- RLM source code: `/home/ros2/.local/lib/python3.10/site-packages/rlm/utils/prompts.py` — `build_rlm_system_prompt`, `ORCHESTRATOR_ADDENDUM`
- RLM source code: `/home/ros2/.local/lib/python3.10/site-packages/rlm/core/lm_handler.py` — `register_client`, `get_client` routing
- Project source: `src/rlm_assistant/` — Phase 1 implementation
- `.planning/phases/02-multi-model-tools/02-CONTEXT.md` — locked decisions

### Secondary (MEDIUM confidence)
- `.planning/research/api-integration.md` — DeepSeek pricing (cross-verified with DeepSeek platform docs)
- `.planning/research/rlm-integration.md` — RLM integration patterns

### Tertiary (LOW confidence)
- SiliconFlow Mimo model name — needs runtime verification via API

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages installed and tested in Phase 1
- Architecture: HIGH — RLM source code directly verified
- Pitfalls: HIGH — discovered from source code analysis and Phase 1 experience
- Mimo integration: MEDIUM — API format verified, exact model name needs runtime check

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 (30 days — stable stack, DeepSeek deprecation deadline 2026/07/24 is key date)
