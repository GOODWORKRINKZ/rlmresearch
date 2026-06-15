"""RLM client wrapper — initializes RLM with DeepSeek multi-model routing.

Pro model handles root-level complex reasoning, Flash model handles sub-calls
(llm_query, rlm_query at depth > 0) for cost optimization. Custom tools
(file ops, command execution, code search) are injected into the REPL.
"""

import logging
import os
from typing import Optional

from rlm import RLM

from rlm_assistant.config import get_settings, Settings
from rlm_assistant.system_prompt import DEV_USER_PROLOGUE
from rlm_assistant.tools import build_custom_tools

logger = logging.getLogger(__name__)

# Module-level singleton
_rlm_instance: Optional[RLM] = None


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

    custom_tools = build_custom_tools(workspace_dir)

    logger.info(
        "Creating RLM instance: pro_model=%s, flash_model=%s, base_url=%s, workspace=%s, persistent=%s",
        settings.deepseek_pro_model,
        settings.deepseek_flash_model,
        settings.deepseek_base_url,
        workspace_dir,
        settings.rlm_persistent,
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs=settings.rlm_backend_kwargs,
        other_backends=["openai"],
        other_backend_kwargs=settings.other_backend_kwargs,
        custom_tools=custom_tools,
        user_prologue=DEV_USER_PROLOGUE,
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

    try:
        logger.debug("Sending message to RLM: %s...", message[:100])
        result = rlm.completion(message)
        response = result.response
        logger.debug("RLM response: %s...", response[:100])
        return response
    except Exception as e:
        logger.error("RLM completion failed: %s", str(e), exc_info=True)
        raise
