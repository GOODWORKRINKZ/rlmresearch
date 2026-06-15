"""RLM client wrapper — initializes RLM with DeepSeek multi-model routing.

Pro model handles root-level complex reasoning, Flash model handles sub-calls
(llm_query, rlm_query at depth > 0) for cost optimization. Custom tools
(file ops, command execution, code search) are injected into the REPL.
"""

import logging
import os
import sys
from typing import Optional

from rlm import RLM

logger = logging.getLogger(__name__)


def _patch_add_context():
    """Monkey-patch LocalREPL to always point 'context' to the latest request.

    Bug in RLM: With persistent=True, add_context() creates context_0, context_1, etc.
    but _restore_scaffold() ALWAYS resets `context = context_0` after every execute_code().
    So the model always sees the FIRST request's data, not the latest.

    Fix: Patch both add_context (to track the latest index) and _restore_scaffold
    (to restore context to the latest, not context_0).
    """
    from rlm.environments.local_repl import LocalREPL

    # -- Patch 1: add_context — track the latest context index --
    _original_add_context = LocalREPL.add_context

    def _fixed_add_context(self, context_payload, context_index=None):
        result = _original_add_context(self, context_payload, context_index)
        # Remember which context index is the latest for this REPL instance
        self._latest_context_index = result
        # Also update locals directly so it's correct before next execute_code
        var_name = f"context_{result}"
        if var_name in self.locals:
            self.locals["context"] = self.locals[var_name]
        return result

    LocalREPL.add_context = _fixed_add_context

    # -- Patch 2: _restore_scaffold — restore context to latest, not context_0 --
    _original_restore_scaffold = LocalREPL._restore_scaffold

    def _fixed_restore_scaffold(self) -> None:
        _original_restore_scaffold(self)
        # After original scaffold restores context=context_0, override to latest
        latest = getattr(self, '_latest_context_index', 0)
        var_name = f"context_{latest}"
        if latest > 0 and var_name in self.locals:
            self.locals["context"] = self.locals[var_name]

    LocalREPL._restore_scaffold = _fixed_restore_scaffold

    sys.stderr.write("[rlm_client] Patched LocalREPL: context always points to latest request\n")
    sys.stderr.flush()
    logger.info("Patched LocalREPL: context always points to latest request")


# Apply patch at import time
_patch_add_context()

from rlm_assistant.config import get_settings, Settings
from rlm_assistant.system_prompt import build_dev_prologue
from rlm_assistant.tools import build_custom_tools

# Module-level singleton
_rlm_instance: Optional[RLM] = None
_custom_tools: Optional[dict] = None


def create_rlm(settings: Optional[Settings] = None, workspace_dir: Optional[str] = None) -> RLM:
    """Create a new RLM instance with DeepSeek backend and custom tools.

    Args:
        settings: Application settings (uses cached default if None).
        workspace_dir: Root directory for tool sandboxing (defaults to RLM_WORKSPACE env or cwd).
    """
    if settings is None:
        settings = get_settings()

    if workspace_dir is None:
        workspace_dir = os.getenv("RLM_WORKSPACE", os.getcwd())

    # RLM_REMOTE=true → RLM has no filesystem access, must use vscode_* tools only
    rlm_remote = os.getenv("RLM_REMOTE", "false").lower() == "true"
    include_local_tools = not rlm_remote

    global _custom_tools
    custom_tools = build_custom_tools(workspace_dir, include_local_tools=include_local_tools)
    _custom_tools = custom_tools

    logger.info(
        "Creating RLM instance: provider=%s, model=%s, base_url=%s, workspace=%s, persistent=%s, remote=%s",
        settings.active_provider,
        settings.active_model_name,
        settings.deepseek_base_url,
        workspace_dir,
        settings.rlm_persistent,
        rlm_remote,
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs=settings.active_backend_kwargs,
        other_backends=["openai"],
        other_backend_kwargs=settings.active_other_backend_kwargs,
        custom_tools=custom_tools,
        user_prologue=build_dev_prologue(include_local_tools=include_local_tools),
        orchestrator=settings.rlm_orchestrator,
        verbose=settings.rlm_verbose,
        persistent=settings.rlm_persistent,
        compaction=settings.rlm_compaction,
        compaction_threshold_pct=settings.rlm_compaction_threshold,
        max_iterations=settings.rlm_max_iterations,
    )

    logger.info("RLM instance created successfully")
    return rlm


def get_rlm() -> RLM:
    """Return the singleton RLM instance (lazy initialization)."""
    global _rlm_instance
    if _rlm_instance is None:
        logger.info("Initializing RLM singleton...")
        _rlm_instance = create_rlm()
    return _rlm_instance


def get_vscode_tool_calls() -> list[dict]:
    """Return pending VS Code tool calls from last RLM execution."""
    if _custom_tools and "_get_vscode_tool_calls" in _custom_tools:
        return _custom_tools["_get_vscode_tool_calls"]["tool"]()
    return []


def get_vscode_id_map() -> dict[str, dict]:
    """Return the tool_call_id → {name, args} mapping from last RLM execution."""
    if _custom_tools and "_get_vscode_id_map" in _custom_tools:
        return _custom_tools["_get_vscode_id_map"]["tool"]()
    return {}


def clear_vscode_tool_calls():
    """Clear pending VS Code tool calls and ID map (call before each RLM run)."""
    if _custom_tools and "_clear_vscode_tool_calls" in _custom_tools:
        _custom_tools["_clear_vscode_tool_calls"]["tool"]()


def chat(message: str, rlm: Optional[RLM] = None) -> str:
    """Send a message to RLM and return the response string.

    Args:
        message: The user message to send
        rlm: Optional RLM instance (uses singleton if not provided)

    Returns:
        Response string from RLM

    Raises:
        Exception: If RLM completion fails
    """
    if rlm is None:
        rlm = get_rlm()

    # Clear VS Code tool call tracker before each run
    clear_vscode_tool_calls()

    try:
        logger.debug("Sending message to RLM: %s...", message[:100])
        result = rlm.completion(message)
        response = result.response
        logger.debug("RLM response: %s...", response[:100])
        return response
    except Exception as e:
        logger.error("RLM completion failed: %s", str(e), exc_info=True)
        raise
