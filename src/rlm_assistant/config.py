"""Configuration module — loads settings from environment variables."""

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file automatically
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_pro_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_PRO_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")))
    deepseek_flash_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"))
    rlm_verbose: bool = field(default_factory=lambda: os.getenv("RLM_VERBOSE", "true").lower() == "true")
    rlm_persistent: bool = True
    rlm_compaction: bool = True
    rlm_compaction_threshold: float = 0.85
    rlm_max_iterations: int = 25
    rlm_orchestrator: bool = field(default_factory=lambda: os.getenv("RLM_ORCHESTRATOR", "true").lower() == "true")
    # Mimo backend settings
    mimo_api_key: str = field(default_factory=lambda: os.getenv("MIMO_API_KEY", ""))
    mimo_base_url: str = field(default_factory=lambda: os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"))
    mimo_model: str = field(default_factory=lambda: os.getenv("MIMO_MODEL", "XiaomiMiMo/MiMo-V2.5-Pro"))
    active_provider: str = field(default_factory=lambda: os.getenv("ACTIVE_PROVIDER", "deepseek"))
    root_provider: str = field(default_factory=lambda: os.getenv("ROOT_PROVIDER", ""))
    sub_provider: str = field(default_factory=lambda: os.getenv("SUB_PROVIDER", ""))
    host: str = field(default_factory=lambda: os.getenv("RLM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("RLM_PORT", "8000")))

    def __post_init__(self):
        if not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Set it in your environment or .env file. "
                "Get your key at https://platform.deepseek.com/api_keys"
            )
        if self.active_provider not in ("deepseek", "mimo"):
            raise ValueError(
                f"ACTIVE_PROVIDER must be 'deepseek' or 'mimo', got '{self.active_provider}'"
            )
        if self.active_provider == "mimo" and not self.mimo_api_key:
            raise ValueError(
                "MIMO_API_KEY is not set but ACTIVE_PROVIDER is 'mimo'. "
                "Set it in your environment or .env file."
            )
        # ROOT_PROVIDER/SUB_PROVIDER override ACTIVE_PROVIDER for mixed routing
        # If not set, fall back to ACTIVE_PROVIDER for both
        if not self.root_provider:
            self.root_provider = self.active_provider
        if not self.sub_provider:
            self.sub_provider = self.active_provider
        # Validate root_provider and sub_provider
        for prov, name in [(self.root_provider, "ROOT_PROVIDER"), (self.sub_provider, "SUB_PROVIDER")]:
            if prov not in ("deepseek", "mimo"):
                raise ValueError(f"{name} must be 'deepseek' or 'mimo', got '{prov}'")
            if prov == "mimo" and not self.mimo_api_key:
                raise ValueError(f"{name} is 'mimo' but MIMO_API_KEY is not set")

    @property
    def rlm_backend_kwargs(self) -> dict:
        """Build backend_kwargs dict for RLM constructor (Pro model)."""
        return {
            "api_key": self.deepseek_api_key,
            "model_name": self.deepseek_pro_model,
            "base_url": self.deepseek_base_url,
        }

    @property
    def other_backend_kwargs(self) -> list[dict]:
        """Build other_backend_kwargs for RLM sub-calls (Flash model)."""
        return [{
            "api_key": self.deepseek_api_key,
            "model_name": self.deepseek_flash_model,
            "base_url": self.deepseek_base_url,
        }]

    @property
    def active_backend_kwargs(self) -> dict:
        """Build backend_kwargs for the root (primary) model."""
        if self.root_provider == "mimo":
            return {
                "api_key": self.mimo_api_key,
                "model_name": self.mimo_model,
                "base_url": self.mimo_base_url,
            }
        return self.rlm_backend_kwargs

    @property
    def active_other_backend_kwargs(self) -> list[dict]:
        """Build other_backend_kwargs for sub-call models.

        RLM only supports ONE additional backend for recursive sub-calls.
        Mimo is available as a custom tool (consult_mimo) instead.
        """
        return [{
            "api_key": self.deepseek_api_key,
            "model_name": self.deepseek_flash_model,
            "base_url": self.deepseek_base_url,
        }]

    @property
    def active_model_name(self) -> str:
        """Return the root provider's primary model name."""
        if self.root_provider == "mimo":
            return self.mimo_model
        return self.deepseek_pro_model

    @property
    def sub_model_name(self) -> str:
        """Return the sub provider's model name."""
        if self.sub_provider == "mimo":
            return self.mimo_model
        return self.deepseek_flash_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    settings = Settings()
    logger.info("Settings loaded: provider=%s, model=%s, base_url=%s",
                settings.active_provider, settings.active_model_name, settings.deepseek_base_url)
    return settings
